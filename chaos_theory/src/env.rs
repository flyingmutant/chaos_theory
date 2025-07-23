// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    cell::RefCell,
    fmt::Debug,
    mem::{replace, take},
    ops::{Deref, DerefMut, RangeBounds as _},
    time::Duration,
};
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(test)]
use crate::tape::TapeCheckpoint;
use crate::{
    Arbitrary, Config, Generator, Source, SourceRaw, Unsigned as _,
    cover::Cover,
    distrib::Biased,
    hash::{hash_bytes, hash_str},
    hash_identity::NoHashSet,
    make::from_fn,
    math::{bitmask, fast_reduce, percent},
    permute::permute,
    rand::{DefaultRand, Rand, Sfc64, Wyrand},
    range::{Range, SizeRange},
    reduce::reduce_tape,
    reproduce_inform,
    tape::Tape,
    tape_event::ScopeKind,
    tape_mutate::MutationCache,
    tape_mutate_crossover::CrossoverCache,
    unwind::{self, PanicInfo},
};

// TODO: would be nice to exceed it *very rarely* to test bigger buffers?
pub(crate) const MAX_SIZE: usize = 65; // close to number of bits in u64, but allows to sometimes overflow buffer of size 64

pub(crate) const TEMPERATURE_DEFAULT: u8 = 128;
pub(crate) const TEMPERATURE_BOUND_EXCLUSIVE: u16 = 256;

const INVALID_CHECKS_MULT: usize = 10;
const INVALID_CHECKS_MIN: usize = 256;
pub(crate) const USE_SEED_PROB: f64 = percent(30);
pub(crate) const USE_SEED_AS_IS_PROB: f64 = percent(15);

// Tweak allows to have several independent index choices inside one scope.
#[repr(u64)]
pub(crate) enum Tweak {
    None = 0,
    SeedChoice = 1,
    FloatSign = 2,
    CharCategory = 3,
    CharRange = 4,
    CharIndex = 5,
    SystemTimeEpoch = 6,
}

const CHECK_UNEXPECTED_NO_PANIC: &str =
    "check unexpectedly did not panic; flaky test or test code change?";

/// Environment and settings for `chaos_theory` magic.
#[derive(Debug)]
pub struct Env {
    log_depth: usize,
    log_verbose: bool,
    budget_remaining: usize,
    temperature: u8,
    seed: u32, // 2^32 seeds is enough for testing
    rng: DefaultRand,
    size_dist: Biased,
    tape_replay: Tape,
    tape_out: Tape,
    scope_id: ScopeId,
    scope_depth: usize,
    scope_depth_manual: usize,
    scope_child_ix: u32,
    scope_version: u32,
    scope_enum_mode: bool,
    cover: Option<Cover>,
    slow: Box<EnvSlow>, // box to make Env smaller
                        // TODO: measure if this Box is useful
}

#[derive(Debug)]
struct EnvSlow {
    log_depth_silent: usize,
    log_depth_default: usize,
    budget: usize,
    check_iters: usize,
    check_time: Duration,
    reduce_time: Duration,
    pretty_print: bool,
    replay_verbose: bool,
    first_example: bool,
    tape_replay_inactive: Vec<Tape>,
    mut_cache: MutationCache,
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Env {
    /// Create a new, default, environment with a fresh random seed.
    ///
    /// See [`Config::env`] for a list of environment variables that affect
    /// the created environment.
    #[must_use]
    pub fn new() -> Self {
        Self::custom().env(true)
    }

    /// Create a config for environment customization.
    pub fn custom() -> Config {
        Config::default()
    }

    /// Obtain an example value of `T`.
    ///
    /// If the `like` reference is provided, example will probably resemble it.
    pub fn example<T: Arbitrary>(&mut self, like: Option<&T>) -> T {
        self.example_of(T::arbitrary(), like)
    }

    /// Obtain an example value from the generator.
    ///
    /// If the `like` reference is provided, example will probably resemble it.
    // TODO: prevent this being callable from `check`
    #[expect(clippy::missing_panics_doc)]
    pub fn example_of<G: Generator>(&mut self, g: G, like: Option<&G::Item>) -> G::Item {
        const EXAMPLE_LABEL: &str = "<example>";
        // The same logic as in `check` iterations: only the first example inherits the `Env` configuration.
        let (seed, tape_replay) = if self.slow.first_example {
            self.slow.first_example = false;
            (self.seed, take(&mut self.tape_replay))
        } else {
            (self.rng.next() as u32, Tape::default())
        };
        let tape = if let Some(like) = like {
            let mut tape_like =
                Self::produce_tape(seed, self.temperature, self.slow.budget, |src| {
                    let _ = src.as_raw().any_of(EXAMPLE_LABEL, &g, Some(like));
                })
                .expect("generator failed to produce seed tape for the provided value");
            let mut rng = DefaultRand::new(u64::from(seed));
            tape_like.mutate(
                &mut rng,
                self.temperature,
                true,
                false,
                true,
                true,
                &mut self.slow.mut_cache,
            );
            tape_like
        } else {
            tape_replay
        };
        let mut src = self.start_from_tape(seed, tape, self.slow.log_depth_silent);
        // TODO: use a version of `filter` here that rolls several times to try to get valid value?
        src.any_of(EXAMPLE_LABEL, g)
    }

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
            unreachable!(
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
                unreachable!("{CHECK_UNEXPECTED_NO_PANIC}");
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

    fn call_prop<T>(prop: impl FnOnce(&mut Source) -> T, src: &mut Source) -> T {
        let v = prop(src);
        // It is in theory possible to produce an invalid tape (by using `catch_unwind` and then continuing,
        // thus using an error tape as a prefix, which results in an invalid tape), but we consider this
        // invalid API usage. Thus, we expect that calling any (normal) prop, the tape passes the validation.
        src.as_mut().tape_out.debug_assert_valid();
        v
    }

    fn call_prop_silent<T>(
        prop: impl FnOnce(&mut Source) -> T,
        src: &mut Source,
    ) -> Result<T, PanicInfo> {
        unwind::catch_silent(|src| Self::call_prop(prop, src), src)
    }
}

impl Env {
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

#[derive(Debug, Default, PartialEq, Eq)]
struct FuzzInput {
    seed: u32,
    tape: Tape,
}

impl FuzzInput {
    fn max_size(&self) -> usize {
        size_of_val(&self.seed) + self.tape.events_max_size()
    }

    fn save(&self, out: &mut [u8]) -> Result<usize, &'static str> {
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

    fn load(&mut self, input: &[u8], validate: bool) -> Result<(), &'static str> {
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
    replay: bool,
    start: Instant,
    valid: usize,
    invalid: usize,
    time_exit: bool,
    cover_done: bool,
    tape: Tape,
    ret: Result<(), PanicInfo>,
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
    #[expect(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub(crate) fn with_params(
        seed: u32,
        temperature: u8,
        budget: usize,
        tape: Option<Tape>,
        cover_depth: usize,
        cover_require: bool,
        check_iters: usize,
        check_time: Duration,
        reduce_time: Duration,
        pretty_print: bool,
        log_depth: usize,
        log_always: bool,
        log_verbose: bool,
        replay_verbose: bool,
    ) -> Self {
        let cover = cover_depth > 0 || cover_require;
        let mut env = Self {
            log_depth: 0,
            log_verbose,
            budget_remaining: budget,
            temperature,
            seed,
            rng: Rand::new(u64::from(seed)),
            size_dist: Biased::new_temperature(temperature, None),
            tape_replay: tape.unwrap_or_default(),
            tape_out: Tape::new(true),
            scope_id: ScopeId::default(),
            scope_depth: 0,
            scope_depth_manual: 0,
            scope_child_ix: 0,
            scope_version: 0,
            scope_enum_mode: false,
            cover: cover.then(|| Cover::new(cover_depth, cover_require)),
            slow: Box::new(EnvSlow {
                log_depth_silent: if log_always { log_depth } else { 0 },
                log_depth_default: log_depth,
                budget,
                check_iters,
                check_time,
                reduce_time,
                pretty_print,
                replay_verbose,
                first_example: true,
                tape_replay_inactive: Vec::default(),
                mut_cache: MutationCache::default(),
            }),
        };
        env.tape_out.reserve_for_replay(&env.tape_replay);
        env
    }

    fn start_from_seed(&mut self, seed: u32, log_depth: usize) -> Source<'_> {
        self.start_from_tape(seed, Tape::default(), log_depth)
    }

    fn init_from_tape(&mut self, seed: u32, tape: Tape, log_depth: usize) {
        // Invariant is that the tape is valid here, but we don't check it to make it possible
        // to debug us producing invalid tapes (or, more commonly, tape validation errors).
        debug_assert!(tape.reuse_at_zero());
        self.seed = seed;
        self.rng = Rand::new(u64::from(seed));
        self.tape_replay = tape;
        self.log_depth = log_depth;
        self.budget_remaining = self.slow.budget;
        self.tape_out.clear();
        self.scope_id = ScopeId::default();
        self.scope_depth = 0;
        self.scope_depth_manual = 0;
        self.scope_child_ix = 0;
        self.scope_version = 0;
        self.scope_enum_mode = false;
    }

    fn start_from_tape(&mut self, seed: u32, tape: Tape, log_depth: usize) -> Source<'_> {
        self.init_from_tape(seed, tape, log_depth);
        Source::new(self)
    }

    pub(crate) fn rng_used(&self) -> bool {
        self.rng != Rand::new(u64::from(self.seed))
    }

    #[cfg(test)]
    pub(crate) fn budget(&self) -> usize {
        self.slow.budget
    }

    #[cfg(test)]
    pub(crate) fn tape_checkpoint(&self) -> TapeCheckpoint {
        self.tape_out.checkpoint()
    }

    #[cfg(test)]
    pub(crate) fn tape_copy_from_checkpoint(
        &self,
        chk: TapeCheckpoint,
        fill_choices: bool,
        copy_meta: bool,
    ) -> Tape {
        self.tape_out
            .copy_from_checkpoint(chk, fill_choices, copy_meta)
    }

    pub(crate) fn should_log(&self) -> bool {
        self.scope_depth - self.scope_depth_manual < self.log_depth
    }

    pub(crate) fn log_value(&self, label: &str, v: &impl Debug) {
        if self.should_log() {
            self.log_value_impl(label, v);
        }
    }

    #[inline(never)]
    fn log_value_impl(&self, label: &str, v: &impl Debug) {
        let empty = "";
        let indent = self.scope_depth * 2;
        if label.is_empty() {
            let label = self.scope_child_ix;
            if self.slow.pretty_print {
                println!("[chaos_theory] {empty:>indent$}${label} = {v:#?}");
            } else {
                println!("[chaos_theory] {empty:>indent$}${label} = {v:?}");
            }
        } else if self.slow.pretty_print {
            println!("[chaos_theory] {empty:>indent$}{label} = {v:#?}");
        } else {
            println!("[chaos_theory] {empty:>indent$}{label} = {v:?}");
        }
    }

    fn log_return(&self, v: &impl Debug) {
        if self.log_verbose && self.should_log() {
            self.log_return_impl(v);
        }
    }

    #[inline(never)]
    fn log_return_impl(&self, v: &impl Debug) {
        let empty = "";
        let indent = self.scope_depth * 2;
        if self.slow.pretty_print {
            println!("[chaos_theory] {empty:>indent$}return {v:#?}");
        } else {
            println!("[chaos_theory] {empty:>indent$}return {v:?}");
        }
    }

    #[expect(clippy::too_many_arguments)]
    fn on_scope_enter(
        &mut self,
        label: &str,
        variant: &str,
        variant_semantic: bool,
        variant_index: usize,
        kind: ScopeKind,
        counter: Option<u32>,
        manual: bool,
    ) {
        if manual && let Some(cover) = &mut self.cover {
            cover.on_scope_enter(label, variant, kind, counter);
        }
        if (manual || self.log_verbose) && self.should_log() {
            self.log_scope_enter_impl(label, variant, variant_semantic, variant_index, counter);
        }
    }

    #[inline(never)]
    #[expect(clippy::collapsible_else_if)]
    fn log_scope_enter_impl(
        &self,
        label: &str,
        variant: &str,
        variant_semantic: bool,
        variant_index: usize,
        counter: Option<u32>,
    ) {
        let empty = "";
        let indent = self.scope_depth * 2;
        if label.is_empty() {
            let label = self.scope_child_ix;
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

    fn on_scope_exit(&mut self, manual: bool) {
        if manual && let Some(cover) = &mut self.cover {
            cover.on_scope_exit();
        }
        // Log nothing for more concise output.
    }

    fn on_effect(&self, label: &str, counter: usize, effect: Effect) {
        if effect != Effect::Success && self.should_log() {
            self.log_effect_impl(label, counter, effect);
        }
    }

    #[inline(never)]
    fn log_effect_impl(&self, label: &str, counter: usize, effect: Effect) {
        let empty = "";
        let indent = self.scope_depth * 2;
        let effect = match effect {
            Effect::Noop => "noop",
            Effect::Change => "change",
            Effect::Success => unreachable!(),
        };
        println!("[chaos_theory] {empty:>indent$}> {effect} {label} #{counter}");
    }

    pub(crate) fn cover_all(&mut self, conditions: &[(&str, bool)]) {
        if let Some(cover) = &mut self.cover {
            cover.cover_all(conditions);
        }
    }

    pub(crate) fn cover_any(&mut self, conditions: &[(&str, bool)]) {
        if let Some(cover) = &mut self.cover {
            cover.cover_any(conditions);
        }
    }
}

impl Env {
    #[doc(hidden)]
    pub fn __start_from_nothing(&mut self, silent: bool) -> Source<'_> {
        let log_depth = if silent {
            self.slow.log_depth_silent
        } else {
            self.slow.log_depth_default
        };
        self.start_from_seed(0, log_depth)
    }

    #[doc(hidden)]
    #[must_use]
    pub fn __at_nothing(&self) -> bool {
        !self.rng_used()
    }

    #[doc(hidden)]
    pub fn __print_input(&self) {
        println!("{}", &self.tape_replay);
    }

    #[doc(hidden)]
    #[must_use]
    pub fn __input(&self) -> (u32, Vec<u8>, usize, bool) {
        let mut buf = vec![0; self.tape_replay.events_max_size()];
        let rem_len = self
            .tape_replay
            .save_events(&mut buf)
            .expect("internal error: failed to save events")
            .len();
        buf.truncate(buf.len() - rem_len);
        (self.seed, buf, self.log_depth, self.slow.pretty_print)
    }

    #[doc(hidden)]
    pub fn __set_input(&mut self, seed: u32, buf: &[u8], log_depth: usize, pretty_print: bool) {
        debug_assert!(self.__at_nothing());
        let mut tape = Tape::default();
        tape.load_events(buf, false, false)
            .expect("internal error: failed to load events");
        self.slow.pretty_print = pretty_print;
        // Note: the tape can be potentially invalid; we don't re-check here.
        self.init_from_tape(seed, tape, log_depth);
    }

    #[doc(hidden)]
    #[must_use]
    pub fn __output(&self) -> Vec<u8> {
        let mut buf = vec![0; self.tape_out.events_max_size()];
        let rem_len = self
            .tape_out
            .save_events(&mut buf)
            .expect("internal error: failed to save events")
            .len();
        buf.truncate(buf.len() - rem_len);
        buf
    }

    #[doc(hidden)]
    pub fn __set_output(&mut self, buf: &[u8]) {
        let mut tape = Tape::new(self.tape_out.has_meta());
        tape.load_events(buf, false, true)
            .expect("internal error: failed to load events");
        // Note: the tape can be potentially invalid; we don't re-check here.
        self.tape_out = tape;
    }
}

/// Result of the [`Source::repeat`] or [`SourceRaw::repeat`](crate::SourceRaw::repeat) step.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
pub enum Effect {
    /// No state change has happened: for example, selected action was inapplicable.
    Noop,
    /// State change (or *possible* state change) without progress: for example, the collection did not grow.
    Change,
    /// Successful state change.
    Success,
}

impl Arbitrary for Effect {
    fn arbitrary() -> impl Generator<Item = Self> {
        from_fn(|src, example| {
            let example_index = example.map(|e| match *e {
                Self::Noop => 0,
                Self::Change => 1,
                Self::Success => 2,
            });
            let (v, _) = src
                .choose(
                    "",
                    example_index,
                    &[Self::Noop, Self::Change, Self::Success],
                )
                .expect("internal error: no effect variants");
            *v
        })
    }
}

// General choice algorithm:
// - we always consume the reuse slot, since we always produce an output one, and they need to match
// - if an example value exists, we try to use it as a "forced" one
// - otherwise, we try to reuse the one from the reuse slot
// - otherwise, we fall back to generating new value

impl Env {
    fn choice_new_size(&mut self, n: usize) -> usize {
        // For recursive data, we might want to half the temperature and set q = 2.0 + 0.25 * depth.
        self.size_dist.sample(&mut self.rng, n)
    }

    pub(crate) fn choose_size(&mut self, r: SizeRange, example: Option<usize>) -> usize {
        let reuse_extra = self
            .tape_replay
            .pop_choice(&mut self.budget_remaining)
            .map(|u| u as usize);
        // When out of budget, always choose minimal size.
        let n = if self.budget_remaining > 0 {
            (r.max - r.min).min(MAX_SIZE) + 1
        } else {
            1
        };
        let example_extra = example
            .filter(|u| r.contains(u))
            .map(|u| u - r.min)
            .filter(|u| *u < n);
        let size_extra = example_extra
            .or_else(|| reuse_extra.filter(|u| *u < n))
            .unwrap_or_else(|| self.choice_new_size(n));
        debug_assert!(size_extra < n);
        let size = r.min + size_extra;
        debug_assert!(r.contains(&size));
        self.budget_remaining = self.budget_remaining.saturating_sub(size.max(1));
        self.tape_out.push_size(size, r);
        size
    }

    fn choice_new_index(&mut self, n: usize, tweak: Tweak) -> usize {
        let tweaked_scope_id = self.scope_id.0.wrapping_add(tweak as u64);
        if self.scope_enum_mode {
            // Get deterministic random value that depends on seed and scope.
            let r = Wyrand::mix(u64::from(self.seed), tweaked_scope_id);
            // Enumerate all possible choices (when the version is incremented).
            let ix = (self.scope_version as usize) % n;
            permute(ix, n, r)
        } else {
            // Get deterministic random value that depends on seed, scope and version.
            let r = Wyrand::mix(
                (u64::from(self.seed) << 32) | u64::from(self.scope_version),
                tweaked_scope_id,
            );
            // Use deterministic random value to transform the bound.
            // By using a bound of 2*n instead of n, we only use swarm testing 50% of the time.
            let m = fast_reduce(r, n.saturating_mul(2));
            if m < n {
                // Do the swarm testing magic.
                permute(self.rng.next_below(m + 1), n, r)
            } else {
                // Just a random choice.
                self.rng.next_below(n)
            }
        }
    }

    pub(crate) fn choose_index(&mut self, n: usize, example: Option<usize>, tweak: Tweak) -> usize {
        debug_assert_ne!(n, 0);
        let reuse = self
            .tape_replay
            .pop_choice(&mut self.budget_remaining)
            .map(|u| u as usize);
        // When out of budget, always choose zero index.
        let n_lim = if self.budget_remaining > 0 { n } else { 1 };
        let example = example.filter(|u| *u < n_lim);
        let index = example
            .or_else(|| reuse.filter(|u| *u < n_lim))
            .unwrap_or_else(|| self.choice_new_index(n_lim, tweak));
        debug_assert!(index < n_lim);
        self.budget_remaining = self.budget_remaining.saturating_sub(n.bit_len().max(1));
        self.tape_out.push_index(index, n);
        index
    }

    pub(crate) fn mark_next_choice_forced(&mut self) {
        self.tape_out.mark_next_choice_forced();
    }

    fn choice_new_bin_size(&mut self, n: usize) -> usize {
        self.size_dist.sample(&mut self.rng, n)
    }

    fn choice_new_value(&mut self, max: u64, bias: bool) -> u64 {
        if bias {
            let total_bits = max.bit_len();
            let use_bits = self.choice_new_bin_size(total_bits + 1);
            let mut w = self.rng.next();
            w &= bitmask::<u64>(use_bits);
            w = w.min(max);
            w
        } else {
            self.rng.next_below_u64(max.saturating_add(1))
        }
    }

    pub(crate) fn choose_value(&mut self, r: Range<u64>, example: Option<u64>, bias: bool) -> u64 {
        // Try to preserve the value bit pattern while ensuring that we fit into required range.
        // This is consistent with our "number-is-a-bit-string" idea.
        fn fit_to_extra(u: Option<u64>, r: Range<u64>, max: u64) -> Option<u64> {
            u.filter(|u| r.contains(u) || r.min == 0).map(|u| {
                let extra = u - r.min;
                if extra <= max {
                    return extra;
                }
                let mut v = extra & bitmask::<u64>(max.bit_len());
                while v > max {
                    // Clear the top bit.
                    v &= !(1 << (v.bit_len() - 1));
                }
                v
            })
        }

        let reuse_extra = self.tape_replay.pop_choice(&mut self.budget_remaining);
        // When out of budget, always choose lower bound.
        let max = if self.budget_remaining > 0 {
            r.max - r.min
        } else {
            0
        };
        let example_extra = fit_to_extra(example, r, max);
        let value_extra = example_extra
            .or_else(|| {
                let reuse = reuse_extra.map(|u| r.min + u);
                fit_to_extra(reuse, r, max)
            })
            .unwrap_or_else(|| self.choice_new_value(max, bias));
        debug_assert!(value_extra <= max);
        let value = r.min + value_extra;
        debug_assert!(r.contains(&value));
        self.budget_remaining = self.budget_remaining.saturating_sub(value.bit_len().max(1));
        self.tape_out.push_value(value, r);
        value
    }

    fn seed_replay_tape<G: Generator>(&mut self, g: &G, seeds: &[G::Item]) -> Option<Tape> {
        self.produce_seed_tape(g, USE_SEED_PROB, seeds)
            .map(|mut t| {
                let use_as_is = self.rng.coinflip(USE_SEED_AS_IS_PROB);
                if !use_as_is {
                    let temperature = self.rng.next() as u8;
                    t.mutate(
                        &mut self.rng,
                        temperature,
                        true,
                        false,
                        true,
                        true,
                        &mut self.slow.mut_cache,
                    );
                    // TODO: crossover would be nice here, too
                }
                t
            })
    }

    pub(crate) fn choose_seed_index(&mut self, prob: f64, seeds: usize) -> Option<usize> {
        if seeds == 0 {
            return None;
        }
        let replay = !self.tape_replay.is_empty();
        if replay && !self.tape_replay.is_void_reuse() {
            // When we operate in the replay mode, and we are not in void reuse,
            // we use the provided tape for everything, so any additional seed tapes are out of question.
            return None;
        }
        let use_seed = self.rng.coinflip(prob);
        if !use_seed {
            return None;
        }
        let seed_ix = self.choice_new_index(seeds, Tweak::SeedChoice);
        Some(seed_ix)
    }

    pub(crate) fn choose_seed<'seeds, T>(
        &mut self,
        prob: f64,
        seeds: &'seeds [T],
    ) -> Option<&'seeds T> {
        self.choose_seed_index(prob, seeds.len())
            .map(|seed_ix| &seeds[seed_ix])
    }

    fn produce_seed_tape<G: Generator>(
        &mut self,
        g: &G,
        prob: f64,
        seeds: &[G::Item],
    ) -> Option<Tape> {
        let seed = self.choose_seed(prob, seeds);
        seed?;
        // Each tape gets its own seed and the same remaining budget.
        Self::produce_tape(
            self.rng.next() as u32,
            self.temperature,
            self.budget_remaining,
            |src| {
                let _ = g.next(src.as_raw(), seed);
            },
        )
    }

    pub(crate) fn produce_tape(
        seed: u32,
        temperature: u8,
        budget: usize,
        prop: impl Fn(&mut Source),
    ) -> Option<Tape> {
        let mut env = Self::custom()
            .with_rng_seed(seed)
            .with_rng_temperature(temperature)
            .with_rng_budget(budget)
            .env(false);
        let mut src = env.start_from_seed(seed, 0);
        // TODO: use a version of `filter` here that rolls several times to try to get valid tape?
        let r = Self::call_prop_silent(prop, &mut src);
        if r.is_ok() {
            let tape = env.tape_out.discard_noop();
            if !tape.is_empty() {
                return Some(tape);
            }
        }
        None
    }

    fn push_replay_tape(&mut self, tape: Tape) {
        debug_assert!(self.tape_replay.is_empty() || self.tape_replay.is_void_reuse());
        self.slow
            .tape_replay_inactive
            .push(take(&mut self.tape_replay));
        self.tape_replay = tape;
    }

    fn pop_replay_tape(&mut self) {
        self.tape_replay = self
            .slow
            .tape_replay_inactive
            .pop()
            .expect("internal error: no inactive tape to pop");
    }
}

// Maybe consider compressing it to 4 bytes: that should be plenty to avoid accidental collisions inside 1 tape.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ScopeId(u64);

impl ScopeId {
    fn combine(self, label: &str, label_ix: u32, variant: &str, variant_ix: u32) -> Self {
        // We prefer to have stable scope IDs based on labels. When there are no
        // manually specified semantic labels, we fall back on indices.
        let l = if label.is_empty() {
            u64::from(label_ix)
        } else {
            hash_str(label)
        };
        let v = if variant.is_empty() {
            u64::from(variant_ix)
        } else {
            hash_str(variant)
        };
        let u = l.rotate_left(32) ^ v;
        Self(Wyrand::mix(self.0, u))
    }
}

pub(crate) struct Scope<'source, S: AsRef<Env> + AsMut<Env>> {
    manual: bool,
    effect: Effect,
    budget_remaining: usize,
    prev_scope_id: ScopeId,
    prev_scope_child_ix: u32,
    prev_scope_version: u32,
    prev_scope_enum_mode: bool,
    src: &'source mut S,
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<Option<Scope<'_, Source>>>() == 40);

impl<'source, S: AsRef<Env> + AsMut<Env>> Scope<'source, S> {
    pub(crate) fn new(
        src: &'source mut S,
        label: &str,
        variant: &str,
        variant_semantic: bool,
        kind: ScopeKind,
        manual: bool,
    ) -> Self {
        Self::new_raw(
            src,
            label,
            variant,
            variant_semantic,
            0,
            kind,
            None,
            manual,
            0,
            false,
        )
    }

    pub(crate) fn new_plain(src: &'source mut S, label: &str, variant: &str) -> Self {
        Self::new(src, label, variant, false, ScopeKind::Plain, false)
    }

    pub(crate) fn new_select_variant(
        src: &'source mut S,
        label: &str,
        variant: &str,
        variant_index: usize,
    ) -> Self {
        Self::new_raw(
            src,
            label,
            variant,
            true,
            variant_index,
            ScopeKind::SelectVariant,
            None,
            true,
            0,
            false,
        )
    }

    pub(crate) fn new_repeat_element(
        src: &'source mut S,
        label: &str,
        counter: u32,
        step_version: u32,
        enum_mode: bool,
    ) -> Self {
        Self::new_raw(
            src,
            label,
            "",
            false,
            0,
            ScopeKind::RepeatElement,
            Some(counter),
            true,
            step_version,
            enum_mode,
        )
    }

    // Version of 0 inherits the current version.
    #[expect(clippy::too_many_arguments)]
    fn new_raw(
        src: &'source mut S,
        label: &str,
        variant: &str,
        variant_semantic: bool,
        variant_index: usize,
        kind: ScopeKind,
        counter: Option<u32>,
        manual: bool,
        mut version: u32,
        mut enum_mode: bool,
    ) -> Self {
        let env = src.as_mut();
        if version == 0 {
            version = env.scope_version;
            enum_mode = env.scope_enum_mode;
        }
        let scope_id = if variant_semantic {
            env.scope_id.combine(
                label,
                env.scope_child_ix,
                variant,
                (variant_index + 1) as u32, // avoid 0: it is reserved for the `else` branch
            )
        } else {
            env.scope_id.combine(label, env.scope_child_ix, "", 0)
        };
        env.on_scope_enter(
            label,
            variant,
            variant_semantic,
            variant_index,
            kind,
            counter,
            manual,
        );
        env.tape_replay.pop_scope_enter(kind);
        env.tape_out.push_scope_enter(scope_id.0, kind);
        env.scope_depth += 1;
        env.scope_depth_manual += usize::from(manual);
        let prev_scope_id = core::mem::replace(&mut env.scope_id, scope_id);
        let prev_scope_child_ix =
            core::mem::take(&mut env.scope_child_ix) + u32::from(kind != ScopeKind::RepeatElement); // Make sure all repeat elements share the same index.
        let prev_scope_version = core::mem::replace(&mut env.scope_version, version);
        let prev_scope_enum_mode = core::mem::replace(&mut env.scope_enum_mode, enum_mode);
        Self {
            manual,
            effect: Effect::Success,
            budget_remaining: env.budget_remaining,
            prev_scope_id,
            prev_scope_child_ix,
            prev_scope_version,
            prev_scope_enum_mode,
            src,
        }
    }

    pub(crate) fn log_return<T: Debug>(&self, v: T) -> T {
        self.as_ref().log_return(&v);
        v
    }

    pub(crate) fn mark_effect(&mut self, label: &str, counter: usize, effect: Effect) {
        self.as_ref().on_effect(label, counter, effect);
        self.effect = effect;
    }
}

impl<S: AsRef<Env> + AsMut<Env>> Deref for Scope<'_, S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        self.src
    }
}

impl<S: AsRef<Env> + AsMut<Env>> DerefMut for Scope<'_, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.src
    }
}

impl<S: AsRef<Env> + AsMut<Env>> Drop for Scope<'_, S> {
    fn drop(&mut self) {
        let env = self.src.as_mut();
        env.scope_depth -= 1;
        env.scope_depth_manual -= usize::from(self.manual);
        env.tape_replay.pop_scope_exit();
        env.tape_out.push_scope_exit(self.effect);
        if self.effect == Effect::Noop {
            // We don't want noop scopes to affect the budget, as that will lead to replay failures.
            env.budget_remaining = self.budget_remaining;
        }
        env.on_scope_exit(self.manual);
        env.scope_id = self.prev_scope_id;
        env.scope_child_ix = self.prev_scope_child_ix;
        env.scope_version = self.prev_scope_version;
        env.scope_enum_mode = self.prev_scope_enum_mode;
    }
}

pub(crate) struct SeedTapeReplayScope<'source, 'env> {
    pub(crate) src: &'source mut SourceRaw<'env>,
    should_pop: bool,
}

impl<'source, 'env> SeedTapeReplayScope<'source, 'env> {
    pub(crate) fn new<G: Generator>(
        src: &'source mut SourceRaw<'env>,
        g: &G,
        seeds: &[G::Item],
    ) -> Self {
        let mut should_pop = false;
        let tape = src.as_mut().seed_replay_tape(g, seeds);
        if let Some(tape) = tape {
            src.as_mut().push_replay_tape(tape);
            should_pop = true;
        }
        Self { src, should_pop }
    }
}

impl Drop for SeedTapeReplayScope<'_, '_> {
    fn drop(&mut self) {
        if self.should_pop {
            self.src.as_mut().pop_replay_tape();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CHECK_ITERS_DEFAULT, assume, check, make, tests::RgbState, vdbg, vprintln};

    #[test]
    fn check_replay_e2e() {
        check(|src| {
            let mut e = Env::custom()
                .with_rng_seed(src.any("seed"))
                .with_rng_temperature(src.any("temperature"))
                .with_rng_budget(src.any("budget"))
                // At least 1 iteration is required, because replay makes one.
                .with_check_iters(
                    src.any_of("check_iters", make::int_in_range(1..=CHECK_ITERS_DEFAULT)),
                )
                // Limit the time spent reducing to speed the test up.
                .with_reduce_time(Duration::from_millis(50))
                .env(false);
            let mut e_state = RgbState::default();
            let e_result = e.check_silent(|src| {
                e_state = RgbState::default();
                e_state.prop_fill(src);
            });

            // Skip the cases where we failed to obtain valid data.
            assume!(
                // `assert_sometimes` would be good here to ensure that property does sometimes fail.
                e_result.ret.is_err()
                    || e_result.valid >= e.slow.check_iters
                    || (e_result.time_exit && e_result.valid > 0)
            );

            vprintln!(src, "replaying...");
            let mut f = Env::custom()
                .with_rng_seed(src.any("replay seed"))
                .with_rng_temperature(src.any("replay temperature"))
                .with_rng_budget(e.slow.budget)
                .with_rng_choices(e_result.tape.as_choices().to_vec())
                .with_check_iters(1)
                .with_reduce_time(Duration::ZERO)
                .env(false);
            let mut f_state = RgbState::default();
            let f_result = f.check_silent(|src| {
                f_state = RgbState::default();
                f_state.prop_fill(src);
            });

            vdbg!(src, (e_result.valid, e_result.invalid, e_result.time_exit));
            vdbg!(src, (f_result.valid, f_result.invalid, f_result.time_exit));
            assert!(!f.rng_used());
            assert_eq!(e_state, f_state);
            assert_eq!(e_result.ret.err(), f_result.ret.err());
            assert_eq!(e_result.tape, f_result.tape);
            assert_eq!(e.budget_remaining, f.budget_remaining);
        });
    }

    #[test]
    fn fuzz_input_roundtrip() {
        check(|src| {
            let seed = src.any("seed");
            let tape = RgbState::default().prop_fill_tape(src, false, false, false);
            let fi = FuzzInput { seed, tape };
            let mut buf = vec![0; fi.max_size()];
            let size = fi.save(&mut buf).unwrap();
            let mut fi_ = FuzzInput::default();
            fi_.load(&buf[..size], false).unwrap();
            assert_eq!(fi, fi_);
        });
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use super::*;
    use crate::Map;
    use core::hint::black_box;
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[bench]
    fn env_new(b: &mut test::Bencher) {
        b.iter(Env::new);
    }

    #[bench]
    fn check_overhead(b: &mut test::Bencher) {
        b.iter(|| {
            let i = black_box(AtomicUsize::new(0));
            Env::custom().with_check_iters(1).env(false).check(|_s| {
                i.fetch_add(black_box(1), Ordering::SeqCst);
            });
            i
        });
    }

    #[bench]
    fn choice_new_index(b: &mut test::Bencher) {
        let mut env = Env::custom().env(false);
        b.iter(|| env.choice_new_index(black_box(10), black_box(Tweak::None)));
    }

    #[bench]
    fn id_combine(b: &mut test::Bencher) {
        b.iter(|| {
            black_box(ScopeId::default()).combine(
                black_box("a"),
                black_box(1),
                black_box("b"),
                black_box(2),
            )
        });
    }

    #[bench]
    fn scope_new_raw(b: &mut test::Bencher) {
        let mut env = Env::custom().with_rng_budget(usize::MAX).env(false);
        let mut src = Source::new(&mut env);
        b.iter(|| {
            let _scope = Scope::new_raw(
                &mut src,
                black_box("a"),
                black_box("b"),
                black_box(false),
                black_box(0),
                black_box(ScopeKind::Plain),
                black_box(None),
                black_box(false),
                black_box(0),
                black_box(false),
            );
        });
    }

    #[bench]
    fn intern_lookup_map(b: &mut test::Bencher) {
        use alloc::sync::Arc;

        let mut env = Env::new();
        let mut m = black_box(Map::default());
        let mut keys = black_box(Vec::new());
        for i in 0..100 {
            let s: Arc<str> = env.example(None);
            keys.push(Arc::clone(&s));
            m.insert(s, i);
        }

        let mut rng = DefaultRand::new(black_box(0));
        b.iter(|| {
            let ix = rng.next_below(keys.len());
            let label = &keys[ix];
            m.get(label)
        });
    }

    #[bench]
    fn intern_append_buf(b: &mut test::Bencher) {
        let mut env = Env::new();
        let mut keys = black_box(Vec::new());
        for _ in 0..100 {
            let s: String = env.example(None);
            keys.push(s);
        }

        let mut rng = DefaultRand::new(black_box(0));
        let mut buf = black_box(String::new());
        buf.reserve(1024 * 1024);
        b.iter(|| {
            let ix = rng.next_below(keys.len());
            let label = &keys[ix];
            buf.push_str(label);
            buf.len()
        });
    }
}
