// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::{format, vec, vec::Vec};
use core::{
    fmt::Debug,
    mem::{replace, take},
};
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    Source, cover::Cover, hash::hash_bytes, hash_identity::NoHashSet, rand::DefaultRand,
    reduce::reduce_tape, reproduce_inform, tape::Tape, tape_mutate::MutationCache,
    tape_mutate_crossover::CrossoverCache, unwind::PanicInfo,
};

use super::{Effect, Env, ReplayMode};

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

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct FuzzInput {
    pub(super) seed: u32,
    pub(super) tape: Tape,
}

impl FuzzInput {
    pub(super) fn max_size(&self) -> usize {
        size_of_val(&self.seed) + self.tape.events_max_size()
    }

    pub(super) fn save(&self, out: &mut [u8]) -> Result<usize, &'static str> {
        Self::save_impl(out, self.seed, &self.tape)
    }

    fn save_impl(out: &mut [u8], seed: u32, tape: &Tape) -> Result<usize, &'static str> {
        let seed_size = size_of_val(&seed);
        if out.len() < seed_size {
            return Err("fuzz input too short");
        }
        out[..seed_size].copy_from_slice(&seed.to_le_bytes());
        let out = &mut out[seed_size..];
        let rem_len = tape.save_events(out)?.len();
        let tape_len = out.len() - rem_len;
        Ok(seed_size + tape_len)
    }

    pub(super) fn load(&mut self, input: &[u8], validate: bool) -> Result<(), &'static str> {
        let seed_size = size_of_val(&self.seed);
        if input.len() < seed_size {
            return Err("fuzz input too short");
        }
        self.seed = u32::from_le_bytes(input[..seed_size].try_into().expect("seed size must be 4"));
        let input = &input[seed_size..];
        let rem_len = self.tape.load_events(input, validate, false)?.len();
        if rem_len != 0 {
            return Err("leftover binary data after tape events");
        }
        Ok(())
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct FuzzState {
    validated_tapes: NoHashSet<u64>,
    last_input: Option<(Vec<u8>, FuzzInput)>,
    last_input_is_effective: bool,
    tape_input: Tape,
    tape_out: Tape,
    mutation_cache: MutationCache,
    crossover_cache: CrossoverCache,
}

impl Default for FuzzState {
    fn default() -> Self {
        Self::new()
    }
}

impl FuzzState {
    #[doc(hidden)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            validated_tapes: NoHashSet::default(),
            last_input: None,
            last_input_is_effective: false,
            tape_input: Tape::default(),
            tape_out: Tape::new(true),
            mutation_cache: MutationCache::default(),
            crossover_cache: CrossoverCache::default(),
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn effective_input(&self) -> Option<&[u8]> {
        self.last_input_is_effective.then(|| {
            self.last_input
                .as_ref()
                .expect("internal error: effective input must exist")
                .0
                .as_slice()
        })
    }

    fn fuzz_cache_validated(&self, input_hash: u64) -> bool {
        self.validated_tapes.contains(&input_hash)
    }

    fn fuzz_cache_mark_validated(&mut self, input_hash: u64) {
        const MAX_VALIDATED_TAPES: usize = 1_000_000; // avoid unbounded growth
        if self.validated_tapes.len() > MAX_VALIDATED_TAPES {
            self.validated_tapes.clear();
        }
        self.validated_tapes.insert(input_hash);
    }

    fn fuzz_cache_take_last_input(&mut self, input: &[u8]) -> Option<(Vec<u8>, FuzzInput)> {
        if self
            .last_input
            .as_ref()
            .is_some_and(|(data, _)| data == input)
        {
            return self.last_input.take();
        }
        None
    }

    fn fuzz_cache_replace_last_input(&mut self, input: Vec<u8>, fi: FuzzInput, effective: bool) {
        self.last_input = Some((input, fi));
        self.last_input_is_effective = effective;
    }

    fn fuzz_cache_take_tape_input(&mut self) -> Tape {
        self.tape_input.clear();
        take(&mut self.tape_input)
    }

    fn fuzz_cache_replace_tape_input(&mut self, tape_input: Tape) {
        debug_assert!(self.tape_input.is_empty());
        self.tape_input = tape_input;
    }

    fn fuzz_cache_take_tape_out(&mut self) -> Tape {
        self.tape_out.clear_for_output();
        take(&mut self.tape_out)
    }

    fn fuzz_cache_replace_tape_out(&mut self, mut tape_out: Tape) {
        debug_assert!(self.tape_out.is_empty());
        tape_out.clear();
        self.tape_out = tape_out;
    }

    fn fuzz_prepare_tape_out(&mut self, tape_input: &mut Tape) {
        // Replay does not need metadata, while recording does. Move its
        // allocation between the two rotating tapes instead of rebuilding it.
        tape_input.move_meta_to(&mut self.tape_out);
        if let Some((_, stale_input)) = &mut self.last_input {
            stale_input.tape.move_meta_to(&mut self.tape_out);
        }
    }

    fn fuzz_load_input(
        &mut self,
        input: &[u8],
        fallback_to_default: bool,
    ) -> Option<(Vec<u8>, FuzzInput)> {
        let fi = self.fuzz_load_input_impl(input);
        if fi.is_none() && fallback_to_default {
            Some((Vec::new(), FuzzInput::default()))
        } else {
            fi
        }
    }

    fn fuzz_load_input_impl(&mut self, input: &[u8]) -> Option<(Vec<u8>, FuzzInput)> {
        if let Some(cached) = self.fuzz_cache_take_last_input(input) {
            return Some(cached);
        }
        let input_hash = hash_bytes(input);
        let validated = self.fuzz_cache_validated(input_hash);
        let mut fi = FuzzInput {
            seed: 0,
            tape: self.fuzz_cache_take_tape_input(),
        };
        fi.load(input, !validated).ok()?;
        if !validated {
            self.fuzz_cache_mark_validated(input_hash);
        }
        Some((Vec::new(), fi))
    }

    fn fuzz_save_last_input_if_fits(
        &mut self,
        out: &mut [u8],
        mut input: Vec<u8>,
        fi: FuzzInput,
    ) -> usize {
        let Ok(size_out) = fi.save(out) else {
            return 0;
        };
        input.clear();
        input.extend_from_slice(&out[..size_out]);
        self.fuzz_cache_mark_validated(hash_bytes(&input));
        self.fuzz_cache_replace_last_input(input, fi, false);
        size_out
    }

    fn fuzz_save_effective_input(&mut self, mut input: Vec<u8>, fi: FuzzInput) {
        let size = if let Ok(size) = fi.save(&mut input) {
            size
        } else {
            input.resize(fi.max_size(), 0);
            fi.save(&mut input)
                .expect("internal error: effective input buffer must be large enough")
        };
        input.truncate(size);
        self.fuzz_cache_replace_last_input(input, fi, true);
    }
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
                let replay_mode = self.shadow_replay_mode(false, Some(res.valid));
                let r = self.run_prop(
                    seed,
                    Tape::default(),
                    replay_mode,
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
            let replay_mode = self.shadow_replay_mode(true, None);
            let r = self.run_prop(
                self.seed,
                tape,
                replay_mode,
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
                        trials += 1;
                    }
                    let replay_mode = self.shadow_replay_mode(true, None);
                    let info = self
                        .run_prop(
                            self.seed,
                            candidate_tape,
                            replay_mode,
                            self.slow.log_depth_silent,
                            true,
                            &mut prop,
                        )
                        .err();
                    (replace(&mut self.tape_out, Tape::new(true)), info)
                });
            res.tape = tape;
            res.ret = Err(info);
        }
        debug_assert!(res.tape.has_meta());
        res
    }

    /// Write seed input for fuzzer.
    ///
    /// See [`crate::fuzz_write_seed`] for the documentation.
    ///
    /// # Errors
    ///
    /// `fuzz_write_seed` fails when valid test case can not be generated or in case of a filesystem error.
    #[expect(clippy::missing_panics_doc)]
    pub fn fuzz_write_seed(
        mut self,
        seed_dir: impl AsRef<Path>,
        prop: impl Fn(&mut Source),
    ) -> Result<(), &'static str> {
        self.slow.check_iters = 1;
        let res = self.check_silent(prop);
        if !(res.ret.is_err()
            || res.valid >= self.slow.check_iters
            || (res.time_exit && res.valid > 0))
        {
            return Err("failed to generate valid test case");
        }
        let fi = FuzzInput {
            seed: self.seed,
            tape: res.tape.discard_noop(), // speed up fuzzing a bit
        };
        let mut buf = vec![0; fi.max_size()];
        let size = fi
            .save(&mut buf)
            .expect("internal error: failed to save seed input");
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| "failed to get current system time since unix epoch")?
            .as_secs();
        let filename = format!("{secs}-{:x}.seed", fi.seed);
        let path = seed_dir.as_ref().join(filename);
        std::fs::create_dir_all(seed_dir)
            .map_err(|_| "failed to create seed directory or one of its parent directories")?;
        std::fs::write(path, &buf[..size]).map_err(|_| "failed to write seed to file")
    }

    #[doc(hidden)]
    #[must_use]
    pub fn fuzz_check<'state>(
        mut self,
        state: &'state mut FuzzState,
        input: &[u8],
        prop: impl Fn(&mut Source),
    ) -> Option<&'state [u8]> {
        state.last_input_is_effective = false;
        let (input_buf, mut fi) = state.fuzz_load_input(input, true)?;
        state.fuzz_prepare_tape_out(&mut fi.tape);
        self.tape_out = state.fuzz_cache_take_tape_out();
        let seed = fi.seed;
        let replay_mode = if self.slow.check_determinism {
            ReplayMode::Strict
        } else {
            ReplayMode::Off
        };
        let r = self.run_prop(
            seed,
            fi.tape,
            replay_mode,
            self.slow.log_depth_silent,
            true,
            &prop,
        );
        if let Err(err) = r {
            if err.invalid_data {
                state.fuzz_cache_replace_tape_input(take(&mut self.tape_replay));
                state.fuzz_cache_replace_tape_out(take(&mut self.tape_out));
                return None;
            }
            // We re-start from the output tape, not the input one:
            // this should not alter the result, but allows to report a more complete tape.
            let tape = replace(&mut self.tape_out, Tape::new(true));
            // Don't try to reduce the input to not trigger a timeout in the fuzzer.
            reproduce_inform(
                self.seed,
                self.temperature,
                self.slow.budget,
                &tape,
                true,
                err.determinism_failure,
                true,
            );
            // Panic for real, unless the test is flaky.
            let _ = self.run_prop(
                self.seed,
                tape,
                replay_mode,
                self.slow.log_depth_default,
                false,
                prop,
            );
        }
        let fi = FuzzInput {
            seed,
            tape: take(&mut self.tape_out),
        };
        state.fuzz_cache_replace_tape_out(take(&mut self.tape_replay));
        state.fuzz_save_effective_input(input_buf, fi);
        state.effective_input()
    }

    #[doc(hidden)]
    #[expect(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn fuzz_mutate(
        self,
        state: &mut FuzzState,
        data: &mut [u8],
        size: usize,
        max_size: usize,
        seed: u32,
        allow_void: bool,
        _mutate_bin: Option<fn(&mut [u8], usize, usize) -> usize>,
    ) -> usize {
        assert!(size <= data.len());
        assert!(max_size <= data.len());
        state.last_input_is_effective = false;
        let Some((input_buf, mut fi)) = state.fuzz_load_input(&data[..size], allow_void) else {
            // Not much we can do.
            return 0;
        };
        state.fuzz_prepare_tape_out(&mut fi.tape);
        let mut rng = DefaultRand::new(u64::from(seed));
        fi.seed = seed;
        fi.tape.mutate(
            &mut rng,
            self.temperature,
            false,
            max_size < size,
            allow_void,
            false,
            &mut state.mutation_cache,
        );
        let out = &mut data[..max_size];
        state.fuzz_save_last_input_if_fits(out, input_buf, fi)
    }

    #[doc(hidden)]
    pub fn fuzz_mutate_crossover(
        self,
        state: &mut FuzzState,
        input: &[u8],
        other: &[u8],
        out: &mut [u8],
        seed: u32,
        allow_void: bool,
    ) -> usize {
        state.last_input_is_effective = false;
        // Note: one of these loads will be uncached, so very slow.
        let Some((mut input_buf, mut fi)) = state.fuzz_load_input(input, allow_void) else {
            // Not much we can do.
            return 0;
        };
        let Some((other_buf, mut other)) = state.fuzz_load_input(other, allow_void) else {
            // Not much we can do.
            return 0;
        };
        state.fuzz_prepare_tape_out(&mut fi.tape);
        state.fuzz_prepare_tape_out(&mut other.tape);
        if other_buf.capacity() > input_buf.capacity() {
            input_buf = other_buf;
        }
        let mut rng = DefaultRand::new(u64::from(seed));
        fi.seed = seed;
        fi.tape = fi.tape.crossover(
            &other.tape,
            &mut rng,
            self.temperature,
            false,
            false,
            &mut state.crossover_cache,
        );
        state.fuzz_save_last_input_if_fits(out, input_buf, fi)
    }
}

#[cfg(test)]
mod tests {
    use crate::Env;
    use core::{cell::Cell, time::Duration};

    #[test]
    fn check_determinism_observe_mismatch_is_advisory_by_default() {
        std::thread_local! {
            static TOGGLE: Cell<bool> = const { Cell::new(false) };
        }

        Env::custom()
            .with_check_iters(1)
            .with_reduce_time(Duration::ZERO)
            .env(false)
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

        Env::custom()
            .with_check_determinism(true)
            .with_check_iters(1)
            .with_reduce_time(Duration::ZERO)
            .env(false)
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

        Env::custom()
            .with_check_determinism(true)
            .with_check_iters(1)
            .with_reduce_time(Duration::ZERO)
            .env(false)
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
        Env::custom()
            .with_check_determinism(true)
            .with_check_iters(1)
            .with_reduce_time(Duration::ZERO)
            .env(false)
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
