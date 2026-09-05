// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    fmt::Debug,
    mem::{replace, take},
};
use std::time::Instant;

use crate::{
    Source, config::reproduce_inform, cover::Cover, reduce::reduce_tape, tape::Tape,
    unwind::PanicInfo,
};

use super::{
    Effect, Env, ReplayMode,
    check_watchdog_std::{CheckWatchdog, HangInfo, WatchGuard},
};

#[path = "fuzz_std.rs"]
mod fuzz_std;

#[cfg(test)]
pub(super) use fuzz_std::FuzzInput;
pub use fuzz_std::FuzzState;

const INVALID_CHECKS_MULT: usize = 10;
const INVALID_CHECKS_MIN: usize = 256;
const CHECK_UNEXPECTED_NO_PANIC: &str =
    "check unexpectedly did not panic; flaky test or test code change?";

pub(super) fn log_value_impl(env: &Env, label: &str, v: &impl Debug) {
    let empty = "";
    let indent = env.scope_depth * 2;
    if label.is_empty() {
        let label = env.scope_child_ix;
        if env.slow.pretty_print {
            println!("[chaos_theory] {empty:>indent$}${label} = {v:#?}");
        } else {
            println!("[chaos_theory] {empty:>indent$}${label} = {v:?}");
        }
    } else if env.slow.pretty_print {
        println!("[chaos_theory] {empty:>indent$}{label} = {v:#?}");
    } else {
        println!("[chaos_theory] {empty:>indent$}{label} = {v:?}");
    }
}

pub(super) fn log_return_impl(env: &Env, v: &impl Debug) {
    let empty = "";
    let indent = env.scope_depth * 2;
    if env.slow.pretty_print {
        println!("[chaos_theory] {empty:>indent$}return {v:#?}");
    } else {
        println!("[chaos_theory] {empty:>indent$}return {v:?}");
    }
}

#[expect(clippy::collapsible_else_if)]
pub(super) fn log_scope_enter_impl(
    env: &Env,
    label: &str,
    variant: &str,
    variant_semantic: bool,
    variant_index: usize,
    counter: Option<u32>,
) {
    let empty = "";
    let indent = env.scope_depth * 2;
    if label.is_empty() {
        let label = env.scope_child_ix;
        if let Some(counter) = counter {
            println!("[chaos_theory] {empty:>indent$}${label} #{counter}:");
        } else if variant_semantic {
            if variant.is_empty() {
                let variant = variant_index;
                println!("[chaos_theory] {empty:>indent$}${label} @{variant}:");
            } else {
                println!("[chaos_theory] {empty:>indent$}${label} {variant}:");
            }
        } else {
            println!("[chaos_theory] {empty:>indent$}${label}:");
        }
    } else {
        if let Some(counter) = counter {
            println!("[chaos_theory] {empty:>indent$}{label} #{counter}:");
        } else if variant_semantic {
            if variant.is_empty() {
                let variant = variant_index;
                println!("[chaos_theory] {empty:>indent$}${label} @{variant}:");
            } else {
                println!("[chaos_theory] {empty:>indent$}{label} {variant}:");
            }
        } else {
            println!("[chaos_theory] {empty:>indent$}{label}:");
        }
    }
}

pub(super) fn log_effect_impl(env: &Env, label: &str, counter: usize, effect: Effect) {
    let empty = "";
    let indent = env.scope_depth * 2;
    let effect = match effect {
        Effect::Noop => "noop",
        Effect::Change => "change",
        Effect::Success => unreachable!(),
    };
    println!("[chaos_theory] {empty:>indent$}> {effect} {label} #{counter}");
}

pub(super) fn print_input(env: &Env) {
    println!("{}", env.tape_replay);
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CheckResult {
    pub(super) replay: bool,
    pub(super) start: Instant,
    pub(super) valid: usize,
    pub(super) invalid: usize,
    pub(super) time_exit: bool,
    pub(super) cover_done: bool,
    pub(super) tape: Tape,
    pub(super) ret: Result<(), PanicInfo>,
}

impl CheckResult {
    fn new() -> Self {
        Self {
            replay: false,
            start: Instant::now(),
            valid: 0,
            invalid: 0,
            time_exit: false,
            cover_done: false,
            tape: Tape::new(true),
            ret: Ok(()),
        }
    }

    fn absorb(&mut self, tape: Tape, ret: Result<(), PanicInfo>, cover_done: bool) {
        debug_assert!(self.ret.is_ok());
        debug_assert!(!self.cover_done || cover_done);
        debug_assert!(tape.has_meta());
        self.tape = tape;
        self.cover_done = cover_done;
        match ret {
            Ok(()) => {
                self.valid += 1;
                self.ret = Ok(());
            }
            Err(info) => {
                if info.invalid_data {
                    self.invalid += 1;
                } else {
                    self.ret = Err(info);
                }
            }
        }
    }
}

impl Env {
    /// Check that property holds (does not panic).
    ///
    /// # Panics
    ///
    /// `check` panics when the property does, or when not enough valid test cases can be generated.
    pub fn check(mut self, prop: impl Fn(&mut Source)) {
        // Use check time for the watchdog timeout for simplicity.
        self.slow.check_watchdog = CheckWatchdog::new(self.slow.check_time);
        let res = self.check_silent(&prop);
        if let Err(err) = res.ret {
            reproduce_inform(
                self.seed,
                self.temperature,
                self.slow.budget,
                &res.tape,
                self.slow.replay_verbose || err.determinism_failure,
                err.determinism_failure,
                false,
            );
            // Panic for real, unless the test is flaky.
            let _watchdog =
                self.watch_hang(HangInfo::new("final failing test case replay", None, None));
            let _ = self.run_prop(
                self.seed,
                res.tape,
                if err.determinism_failure {
                    ReplayMode::Strict
                } else {
                    ReplayMode::Off
                },
                self.slow.log_depth_default,
                false,
                prop,
            );
            panic!(
                "{CHECK_UNEXPECTED_NO_PANIC}\nPanic we were trying to reproduce ({}:{}): {}",
                err.file, err.line, err.message
            );
        }
        let valid = res.valid;
        let total = res.valid + res.invalid;
        let elapsed = res.start.elapsed();
        if res.replay {
            // If multiple `check` calls are in one test, and non-first one had failed,
            // then during replay these messages will show up for all checks before the failing one.
            eprintln!("[chaos_theory] {CHECK_UNEXPECTED_NO_PANIC}");
        } else if res.valid >= self.slow.check_iters || (res.time_exit && res.valid > 0) {
            let have_cover = self.cover.is_some();
            let cover = if have_cover {
                if res.cover_done {
                    ", cover finished"
                } else {
                    ", cover not finished"
                }
            } else {
                ""
            };
            if res.time_exit {
                let limit = self.slow.check_time;
                eprintln!(
                    "[chaos_theory] ~OK, passed {valid} tests ({elapsed:?} elapsed with time limit of {limit:?}{cover})"
                );
            } else {
                eprintln!("[chaos_theory] OK, passed {valid} tests ({elapsed:?}{cover})");
            }
        } else {
            eprintln!(
                "[chaos_theory] only generated {valid} valid tests from {total} total ({elapsed:?})"
            );
            let _watchdog =
                self.watch_hang(HangInfo::new("last invalid test case replay", None, None));
            let mut src = self.start_from_tape(self.seed, res.tape, self.slow.log_depth_default);
            // Panic with last invalid data (to help debug the issue), unless the test is flaky.
            Self::call_prop(prop, &mut src);
            panic!("{CHECK_UNEXPECTED_NO_PANIC}");
        }
    }

    // Returns the last (either successful or failing) tape.
    pub(crate) fn check_silent(&mut self, mut prop: impl FnMut(&mut Source)) -> CheckResult {
        let mut res = CheckResult::new();
        if self.tape_replay.is_empty() {
            let checks = self.slow.check_iters;
            let invalid_checks = checks
                .saturating_mul(INVALID_CHECKS_MULT)
                .max(INVALID_CHECKS_MIN);
            // Importantly, CHAOS_THEORY_RNG_SEED can directly specify the seed for the first iteration.
            let base_seed = self.seed;
            while res.valid < checks
                || (self.cover.as_ref().is_some_and(Cover::require) && !res.cover_done)
            {
                if res.invalid >= invalid_checks {
                    break;
                }
                let i = res.valid + res.invalid;
                let elapsed = res.start.elapsed();
                if elapsed > self.slow.check_time {
                    res.time_exit = true;
                    break;
                }
                let seed = base_seed.wrapping_add(i as u32);
                if self.slow.log_depth_silent > 0 {
                    let valid = res.valid;
                    let invalid = res.invalid;
                    eprintln!(
                        "[chaos_theory/iters/{i}] starting check iteration (seed: {seed:08x}, done: {valid} valid, {invalid} invalid)"
                    );
                }
                let _watchdog =
                    self.watch_hang(HangInfo::new("check iteration", Some(i), Some(seed)));
                let r = self.run_prop(
                    seed,
                    Tape::default(),
                    self.shadow_replay_mode(false, Some(res.valid)),
                    self.slow.log_depth_silent,
                    true,
                    &mut prop,
                );
                // TODO: replacing the `tape_out` with new one does not allow for allocation reuse
                res.absorb(
                    replace(&mut self.tape_out, Tape::new(true)),
                    r,
                    self.cover.as_ref().is_some_and(Cover::done),
                );
                if self.budget_remaining == 0 {
                    eprintln!(
                        "[chaos_theory] ran out of RNG budget at check {i}/{checks}, review your code or consider increasing the budget"
                    );
                }
                if res.ret.is_err() {
                    break;
                }
            }
        } else {
            res.replay = true;
            let tape = take(&mut self.tape_replay);
            let _watchdog = self.watch_hang(HangInfo::new("replayed test case", None, None));
            let r = self.run_prop(
                self.seed,
                tape,
                self.shadow_replay_mode(true, None),
                self.slow.log_depth_silent,
                true,
                &mut prop,
            );
            res.absorb(
                replace(&mut self.tape_out, Tape::new(true)),
                r,
                self.cover.as_ref().is_some_and(Cover::done),
            );
        }
        // TODO: add stage that mutates generated tapes (only large-scale, structural mutations that make sense)
        if let Err(info) = res.ret {
            let mut trials = 0;
            let (tape, info) =
                reduce_tape(res.tape, info, self.slow.reduce_time, |candidate_tape| {
                    if self.slow.log_depth_silent > 0 {
                        eprintln!(
                            "[chaos_theory/trials/{trials}] starting test case reduction trial"
                        );
                    }
                    let _watchdog =
                        self.watch_hang(HangInfo::new("reduction trial", Some(trials), None));
                    let info = self
                        .run_prop(
                            self.seed,
                            candidate_tape,
                            self.shadow_replay_mode(true, None),
                            self.slow.log_depth_silent,
                            true,
                            &mut prop,
                        )
                        .err();
                    trials += 1;
                    (replace(&mut self.tape_out, Tape::new(true)), info)
                });
            res.tape = tape;
            res.ret = Err(info);
        }
        debug_assert!(res.tape.has_meta());
        res
    }

    fn watch_hang(&self, info: HangInfo) -> Option<WatchGuard> {
        self.slow
            .check_watchdog
            .as_ref()
            .map(|watchdog| watchdog.start(info))
    }
}

#[cfg(test)]
mod tests {
    use crate::{Effect, Env, make};
    use core::{cell::Cell, time::Duration};

    #[test]
    fn check_trivializes_nested_vec_to_minimum() {
        const N: usize = 1000;
        let choices = core::iter::once(0)
            .chain([1, 0, 13, 0, 13, 0, 13].into_iter().cycle().take(7 * N))
            .collect();
        let mut actual = Vec::new();
        let mut calls = 0;
        let result = Env::builder()
            .with_rng_choices(choices)
            .build()
            .check_silent(|src| {
                calls += 1;
                actual = src.any_of(
                    "v",
                    make::vec_n(make::vec_n(make::arbitrary::<u64>(), 2..=3), N..=N),
                );
                assert!(actual.is_empty());
            });

        assert!(result.ret.is_err());
        assert_eq!(actual, vec![vec![0, 0]; N]);
        assert!(calls <= 8, "trivialization took {calls} property calls");
    }

    #[test]
    fn check_reduces_nested_stateful_case() {
        type Op = Result<(u64, (i64, i64)), (i64, i64)>;

        // `Err` arms the state and `Ok` fires. Keeping fire as the lower variant makes
        // the minimal failing sequence oppose the reducer's preferred sort order.
        let mut actual: Vec<Op> = Vec::new();
        let result = Env::builder()
            .with_check_iters(1)
            .build()
            .check_silent(|src| {
                actual.clear();
                let mut armed = false;
                let mut failed = false;
                src.repeat_n("ops", 2.., |src| {
                    src.select("op", &["fire", "arm"], |src, variant, _ix| {
                        let op = match variant {
                            "arm" => {
                                if armed {
                                    return Effect::Noop;
                                }
                                armed = true;
                                Err(src.any("payload"))
                            }
                            "fire" => {
                                if !armed {
                                    return Effect::Noop;
                                }
                                failed = true;
                                let power: u64 = src.any("power");
                                Ok((power.max(1), src.any("payload")))
                            }
                            _ => unreachable!(),
                        };
                        actual.push(op);
                        Effect::Success
                    })
                });
                assert!(!failed);
            });

        assert!(result.ret.is_err());
        assert_eq!(actual, [Err((0, 0)), Ok((1, (0, 0)))]);
    }

    #[test]
    fn check_determinism_observe_mismatch_is_advisory_by_default() {
        std::thread_local! {
            static TOGGLE: Cell<bool> = const { Cell::new(false) };
        }

        Env::builder()
            .with_check_iters(1)
            .with_reduce_time(Duration::ZERO)
            .build()
            .check(|src| {
                let value = TOGGLE.with(|toggle| {
                    let value = u64::from(toggle.get());
                    toggle.set(!toggle.get());
                    value
                });
                src.observe("toggle", value);
            });
    }

    #[test]
    #[should_panic(expected = "determinism check failed: observation mismatch for `toggle`")]
    fn check_determinism_observe_mismatch() {
        std::thread_local! {
            static TOGGLE: Cell<bool> = const { Cell::new(false) };
        }

        Env::builder()
            .with_check_determinism(true)
            .with_check_iters(1)
            .with_reduce_time(Duration::ZERO)
            .build()
            .check(|src| {
                let value = TOGGLE.with(|toggle| {
                    let value = u64::from(toggle.get());
                    toggle.set(!toggle.get());
                    value
                });
                src.observe("toggle", value);
            });
    }

    #[test]
    #[should_panic(expected = "determinism check failed: observation mismatch for `counter`")]
    fn check_determinism_failure_is_preserved_when_rerun_is_flaky() {
        std::thread_local! {
            static COUNTER: Cell<u32> = const { Cell::new(0) };
        }

        Env::builder()
            .with_check_determinism(true)
            .with_check_iters(1)
            .with_reduce_time(Duration::ZERO)
            .build()
            .check(|src| {
                let value = COUNTER.with(|counter| {
                    let next = counter.get();
                    counter.set(next + 1);
                    u64::from(next != 0)
                });
                src.observe("counter", value);
            });
    }

    #[test]
    #[should_panic(expected = "self-replay failed after a successful first pass: boom")]
    fn check_determinism_second_pass_failure() {
        std::thread_local! {
            static CALLS: Cell<u32> = const { Cell::new(0) };
        }

        CALLS.with(|calls| calls.set(0));
        Env::builder()
            .with_check_determinism(true)
            .with_check_iters(1)
            .with_reduce_time(Duration::ZERO)
            .build()
            .check(|_src| {
                let call = CALLS.with(|calls| {
                    let call = calls.get() + 1;
                    calls.set(call);
                    call
                });
                assert!(call != 2, "boom");
            });
    }
}
