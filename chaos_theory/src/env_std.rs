// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::{format, vec, vec::Vec};
use core::{
    cell::RefCell,
    fmt::Debug,
    mem::{replace, take},
};
use std::{
    path::Path,
    thread_local,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    Source,
    cover::Cover,
    hash::hash_bytes,
    hash_identity::NoHashSet,
    rand::{DefaultRand, Rand, Sfc64},
    reduce::reduce_tape,
    reproduce_inform,
    tape::Tape,
    tape_mutate::MutationCache,
    tape_mutate_crossover::CrossoverCache,
    unwind::PanicInfo,
};

use super::{Effect, Env};

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
    println!("{}", &env.tape_replay);
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
                self.slow.replay_verbose,
                false,
            );
            let mut src = self.start_from_tape(self.seed, res.tape, self.slow.log_depth_default);
            // Panic for real, unless the test is flaky.
            Self::call_prop(prop, &mut src);
            panic!(
                "{CHECK_UNEXPECTED_NO_PANIC}\nPanic we were trying to reproduce ({}:{}): {}",
                err.file, err.line, err.message
            );
        } else {
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
                let mut src =
                    self.start_from_tape(self.seed, res.tape, self.slow.log_depth_default);
                // Panic with last invalid data (to help debug the issue), unless the test is flaky.
                Self::call_prop(prop, &mut src);
                panic!("{CHECK_UNEXPECTED_NO_PANIC}");
            }
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
            // Use a different PRNG for the seed sequence to avoid seed sequence cycles.
            let mut seed_gen = Rand::<Sfc64>::new(u64::from(self.seed));
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
                if self.slow.log_depth_silent > 0 {
                    let valid = res.valid;
                    let invalid = res.invalid;
                    eprintln!(
                        "[chaos_theory/iters/{i}] starting check iteration (done: {valid} valid, {invalid} invalid)"
                    );
                }
                let r = {
                    let seed = seed_gen.next() as u32;
                    let mut src = self.start_from_seed(seed, self.slow.log_depth_silent);
                    Self::call_prop_silent(&mut prop, &mut src)
                };
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
            let mut src = self.start_from_tape(self.seed, tape, self.slow.log_depth_silent);
            let r = Self::call_prop_silent(&mut prop, &mut src);
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
                    let mut src =
                        self.start_from_tape(self.seed, candidate_tape, self.slow.log_depth_silent);
                    let info = Self::call_prop_silent(&mut prop, &mut src).err();
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

    thread_local! {
        // Avoid repeated tape validations.
        static VALIDATED_TAPES: RefCell<NoHashSet<u64>> = RefCell::new(NoHashSet::default());
        // Optimize constant save/load overhead caused by custom mutator interface.
        static LAST_FUZZ_INPUT: RefCell<Option<(Vec<u8>, FuzzInput)>> = const { RefCell::new(None) };
        // Reuse input tape allocation.
        static TAPE_INPUT: RefCell<Tape> = RefCell::new(Tape::default());
        // Reuse output tape allocation.
        static TAPE_OUT: RefCell<Tape> = RefCell::new(Tape::new(true));
        // Reuse mutation cache allocations between mutations.
        static MUTATION_CACHE: RefCell<MutationCache> = RefCell::new(MutationCache::default());
        // Reuse crossover cache allocations between mutations.
        static CROSSOVER_CACHE: RefCell<CrossoverCache> = RefCell::new(CrossoverCache::default());
    }

    fn fuzz_cache_validated(input_hash: u64) -> bool {
        Self::VALIDATED_TAPES.with_borrow(|s| s.contains(&input_hash))
    }

    fn fuzz_cache_mark_validated(input_hash: u64) {
        const MAX_VALIDATED_TAPES: usize = 1_000_000; // avoid unbounded growth
        Self::VALIDATED_TAPES.with_borrow_mut(|s| {
            if s.len() > MAX_VALIDATED_TAPES {
                s.clear();
            }
            s.insert(input_hash);
        });
    }

    fn fuzz_cache_take_last_input(input: &[u8]) -> Option<FuzzInput> {
        Self::LAST_FUZZ_INPUT.with_borrow_mut(|cache| {
            if let Some((data, _input)) = cache
                && data == input
            {
                return take(cache).map(|c| c.1);
            }
            None
        })
    }

    fn fuzz_cache_replace_last_input(input: Vec<u8>, fi: FuzzInput) {
        // Don't try to reuse the input allocation to simplify things a bit.
        Self::LAST_FUZZ_INPUT.with_borrow_mut(|cache| cache.replace((input, fi)));
    }

    fn fuzz_cache_take_tape_input() -> Tape {
        Self::TAPE_INPUT.with_borrow_mut(|input| {
            input.clear();
            take(input)
        })
    }

    fn fuzz_cache_replace_tape_input(tape_input: Tape) {
        Self::TAPE_INPUT.with_borrow_mut(|input| {
            debug_assert!(input.is_empty());
            let _ = replace(input, tape_input);
        });
    }

    fn fuzz_cache_take_tape_out() -> Tape {
        Self::TAPE_OUT.with_borrow_mut(|out| {
            out.clear();
            replace(out, Tape::new(true))
        })
    }

    fn fuzz_cache_replace_tape_out(tape_out: Tape) {
        Self::TAPE_OUT.with_borrow_mut(|out| {
            debug_assert!(out.is_empty());
            let _ = replace(out, tape_out);
        });
    }

    fn fuzz_load_input(input: &[u8], fallback_to_default: bool) -> Option<FuzzInput> {
        let fi = Self::fuzz_load_input_impl(input);
        if fi.is_none() && fallback_to_default {
            Some(FuzzInput::default())
        } else {
            fi
        }
    }

    fn fuzz_load_input_impl(input: &[u8]) -> Option<FuzzInput> {
        let cached_fi = Self::fuzz_cache_take_last_input(input);
        if cached_fi.is_some() {
            return cached_fi;
        }
        let input_hash = hash_bytes(input);
        let validated = Self::fuzz_cache_validated(input_hash);
        let mut fi = FuzzInput {
            seed: 0,
            tape: Self::fuzz_cache_take_tape_input(),
        };
        fi.load(input, !validated).ok()?;
        if !validated {
            Self::fuzz_cache_mark_validated(input_hash);
        }
        Some(fi)
    }

    fn fuzz_save_last_input_if_fits(out: &mut [u8], fi: FuzzInput) -> usize {
        let Ok(size_out) = fi.save(out) else {
            return 0;
        };
        let input = &out[..size_out];
        Self::fuzz_cache_mark_validated(hash_bytes(input));
        Self::fuzz_cache_replace_last_input(input.to_vec(), fi);
        size_out
    }

    /// Check that property holds (does not panic) on fuzzer-provided input.
    ///
    /// See [`crate::fuzz_check`] for the documentation.
    ///
    /// # Panics
    ///
    /// `fuzz_check` panics when the property does.
    #[must_use]
    pub fn fuzz_check(
        mut self,
        input: &[u8],
        out: Option<(&mut [u8], &mut usize)>,
        prop: impl Fn(&mut Source),
    ) -> bool {
        let Some(fi) = Self::fuzz_load_input(input, out.is_some()) else {
            return false;
        };
        self.tape_out = Self::fuzz_cache_take_tape_out();
        let mut src = self.start_from_tape(fi.seed, fi.tape, self.slow.log_depth_silent);
        let r = Self::call_prop_silent(&prop, &mut src);
        if let Some((out, out_size)) = out {
            assert!(out.len() >= input.len());
            if let Ok(out_size_used) = FuzzInput::save_impl(out, fi.seed, &self.tape_out) {
                *out_size = out_size_used;
            } else {
                // Fall back to input if output does not fit.
                out[..input.len()].copy_from_slice(input);
                *out_size = input.len();
            }
        }
        if let Err(err) = r {
            if err.invalid_data {
                return false;
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
                true,
            );
            let mut s = self.start_from_tape(self.seed, tape, self.slow.log_depth_default);
            // Panic for real, unless the test is flaky.
            prop(&mut s);
        } else {
            Self::fuzz_cache_replace_tape_input(self.tape_replay);
            Self::fuzz_cache_replace_tape_out(self.tape_out);
        }
        true
    }

    /// Mutate fuzzer input.
    ///
    /// See [`crate::fuzz_mutate`] for the documentation.
    #[expect(clippy::missing_panics_doc, clippy::type_complexity)]
    pub fn fuzz_mutate(
        self,
        data: &mut [u8],
        size: usize,
        max_size: usize,
        seed: u32,
        allow_void: bool,
        _mutate_bin: Option<fn(&mut [u8], usize, usize) -> usize>,
    ) -> usize {
        assert!(size <= data.len());
        assert!(max_size <= data.len());
        let Some(mut fi) = Self::fuzz_load_input(&data[..size], allow_void) else {
            // Not much we can do.
            return 0;
        };
        let mut rng = DefaultRand::new(u64::from(seed));
        fi.seed = seed;
        Self::MUTATION_CACHE.with_borrow_mut(|cache| {
            fi.tape.mutate(
                &mut rng,
                self.temperature,
                false,
                max_size < size,
                allow_void,
                false,
                cache,
            );
        });
        let out = &mut data[..max_size];
        Self::fuzz_save_last_input_if_fits(out, fi)
    }

    /// Cross-over two fuzzer inputs.
    ///
    /// See [`crate::fuzz_mutate_crossover`] for the documentation.
    pub fn fuzz_mutate_crossover(
        self,
        input: &[u8],
        other: &[u8],
        out: &mut [u8],
        seed: u32,
        allow_void: bool,
    ) -> usize {
        // Note: one of these loads will be uncached, so very slow.
        let Some(mut fi) = Self::fuzz_load_input(input, allow_void) else {
            // Not much we can do.
            return 0;
        };
        let Some(other) = Self::fuzz_load_input(other, allow_void) else {
            // Not much we can do.
            return 0;
        };
        let mut rng = DefaultRand::new(u64::from(seed));
        fi.seed = seed;
        fi.tape = Self::CROSSOVER_CACHE.with_borrow_mut(|cache| {
            fi.tape
                .crossover(&other.tape, &mut rng, self.temperature, false, false, cache)
        });
        Self::fuzz_save_last_input_if_fits(out, fi)
    }
}
