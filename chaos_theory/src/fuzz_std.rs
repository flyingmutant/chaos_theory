// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::vec::Vec;
use core::mem::{replace, take};

use crate::{
    Source, config::reproduce_inform, hash::hash_bytes, hash_identity::NoHashSet,
    rand::DefaultRand, tape::Tape, tape_mutate::MutationCache,
    tape_mutate_crossover::CrossoverCache,
};

use super::super::{Env, ReplayMode};

#[derive(Debug, Default, PartialEq, Eq)]
pub(in crate::env) struct FuzzInput {
    pub(in crate::env) seed: u32,
    pub(in crate::env) tape: Tape,
}

impl FuzzInput {
    pub(in crate::env) fn max_size(&self) -> usize {
        size_of_val(&self.seed) + self.tape.events_max_size()
    }

    pub(in crate::env) fn save(&self, out: &mut [u8]) -> Result<usize, &'static str> {
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

    pub(in crate::env) fn load(
        &mut self,
        input: &[u8],
        validate: bool,
    ) -> Result<(), &'static str> {
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

impl Env {
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
