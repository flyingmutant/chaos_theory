// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::{boxed::Box, format, vec, vec::Vec};
use core::{
    fmt::Debug,
    mem::take,
    ops::{Deref, DerefMut, RangeBounds as _},
    time::Duration,
};

#[cfg(test)]
use crate::tape::TapeCheckpoint;
#[cfg(feature = "std")]
use crate::unwind::catch_silent_info;
use crate::{
    Arbitrary, Config, Generator, Source, SourceEx, Unsigned as _,
    cover::Cover,
    distrib::Biased,
    hash::hash_str,
    make::from_next,
    math::{bitmask, fast_reduce, mul_add, percent},
    permute::permute,
    rand::{DefaultRand, Rand, Wyrand},
    range::{Range, SizeRange},
    tape::Tape,
    tape_event::{Event, ScopeKind},
    tape_mutate::MutationCache,
    unwind::{DETERMINISM_FAILED_PREFIX, PanicInfo, panic_determinism},
    util::DebugOutputGuard,
};

#[cfg(feature = "std")]
#[path = "check_watchdog_std.rs"]
mod check_watchdog_std;

#[cfg(feature = "std")]
#[path = "env_std.rs"]
mod std_impl;

#[cfg(feature = "std")]
#[doc(hidden)]
pub use std_impl::FuzzState;

#[cfg(all(test, feature = "std"))]
use std_impl::FuzzInput;

#[cfg(not(feature = "std"))]
fn log_value_impl(_env: &Env, _label: &str, _v: &impl Debug) {}

#[cfg(feature = "std")]
use std_impl::log_value_impl;

#[cfg(not(feature = "std"))]
fn log_return_impl(_env: &Env, _v: &impl Debug) {}

#[cfg(feature = "std")]
use std_impl::log_return_impl;

#[cfg(not(feature = "std"))]
fn log_scope_enter_impl(
    _env: &Env,
    _label: &str,
    _variant: &str,
    _variant_semantic: bool,
    _variant_index: usize,
    _counter: Option<u32>,
) {
}

#[cfg(feature = "std")]
use std_impl::log_scope_enter_impl;

#[cfg(not(feature = "std"))]
fn log_effect_impl(_env: &Env, _label: &str, _counter: usize, _effect: Effect) {}

#[cfg(feature = "std")]
use std_impl::log_effect_impl;

#[cfg(not(feature = "std"))]
fn print_input(_env: &Env) {}

#[cfg(feature = "std")]
use std_impl::print_input;

// TODO: would be nice to exceed it *very rarely* to test bigger buffers?
pub(crate) const MAX_SIZE: usize = 65; // close to number of bits in u64, but allows to sometimes overflow buffer of size 64

pub(crate) const TEMPERATURE_DEFAULT: u8 = 128;
pub(crate) const TEMPERATURE_BOUND_EXCLUSIVE: u16 = 256;

pub(crate) const USE_SEED_PROB: f64 = percent(30);
pub(crate) const USE_SEED_AS_IS_PROB: f64 = percent(15);

// Tweak allows to have several independent swarm choices inside one scope.
#[repr(u64)]
pub(crate) enum Tweak {
    None = 0,
    SeedChoice = 1,
    IntSign = 2,
    FloatSign = 3,
    CharCategory = 4,
    CharRange = 5,
    CharIndex = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplayMode {
    Off,
    Lax { diverged: bool },
    Strict,
}

/// Structural pseudo-random generation and property-checking environment.
#[derive(Debug)]
pub struct Env {
    log_depth: usize,
    log_verbose: bool,
    budget_remaining: usize,
    temperature: u8,
    seed: u32, // 2^32 seeds is enough for testing
    rng: DefaultRand,
    size_dist: Biased,
    replay_mode: ReplayMode,
    tape_replay: Tape,
    tape_out: Tape,
    repeat_noop_ixs: Vec<u32>,
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
#[expect(clippy::struct_excessive_bools)]
struct EnvSlow {
    log_depth_silent: usize,
    log_depth_default: usize,
    budget: usize,
    check_iters: usize,
    check_determinism: bool,
    check_time: Duration,
    reduce_time: Duration,
    pretty_print: bool,
    replay_verbose: bool,
    first_generation: bool,
    tape_replay_inactive: Vec<Tape>,
    mut_cache: MutationCache,
    #[cfg(feature = "std")]
    check_watchdog: Option<check_watchdog_std::CheckWatchdog>,
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Env {
    /// Create a default environment.
    ///
    /// This does not read environment variables. Use [`Env::builder`] with
    /// [`Config::with_env_vars`] to enable them.
    #[must_use]
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Create a builder for a customized environment.
    pub fn builder() -> Config {
        Config::default()
    }

    // TODO: prevent generation methods being callable from `check`

    /// Generate a value of `T`.
    pub fn generate<T: Arbitrary>(&mut self) -> T {
        self.generate_with(T::arbitrary())
    }

    /// Generate a value using the provided generator.
    pub fn generate_with<G: Generator>(&mut self, g: G) -> G::Item {
        self.generate_with_example(g, None)
    }

    pub(crate) fn generate_with_example<G: Generator>(
        &mut self,
        g: G,
        example: Option<&G::Item>,
    ) -> G::Item {
        const GENERATE_LABEL: &str = "<generate>";
        // The same logic as in `check` iterations: only the first generated value
        // inherits the `Env` configuration.
        let (seed, tape_replay) = if self.slow.first_generation {
            self.slow.first_generation = false;
            (self.seed, take(&mut self.tape_replay))
        } else {
            (self.rng.next() as u32, Tape::default())
        };
        let tape = if let Some(example) = example {
            let mut tape_like =
                Self::produce_tape(seed, self.temperature, self.slow.budget, |src| {
                    let _ = src.as_ex().any_of(GENERATE_LABEL, &g, Some(example));
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
        let _debug_output_guard = DebugOutputGuard::new(src.as_ref().should_log());
        // TODO: use a version of `filter` here that rolls several times to try to get valid value?
        src.any_of(GENERATE_LABEL, g)
    }

    fn call_prop<T>(prop: impl FnOnce(&mut Source) -> T, src: &mut Source) -> T {
        let _debug_output_guard = DebugOutputGuard::new(src.as_ref().should_log());
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
        silent: bool,
    ) -> Result<T, PanicInfo> {
        if silent {
            #[cfg(feature = "std")]
            return catch_silent_info(|src| Self::call_prop(prop, src), src);
            #[cfg(not(feature = "std"))]
            return Ok(Self::call_prop(prop, src));
        }
        Ok(Self::call_prop(prop, src))
    }

    fn run_prop(
        &mut self,
        seed: u32,
        tape: Tape,
        replay_mode: ReplayMode,
        log_depth: usize,
        silent: bool,
        mut prop: impl FnMut(&mut Source),
    ) -> Result<(), PanicInfo> {
        let ret = {
            let mut src = self.start_from_tape(seed, tape, log_depth);
            Self::call_prop_silent(&mut prop, &mut src, silent)
        };
        if ret.is_err() || replay_mode == ReplayMode::Off {
            return ret;
        }

        let tape_out = core::mem::replace(&mut self.tape_out, Tape::new(true));
        let replay_ret = {
            self.init_from_tape(seed, tape_out.clone(), log_depth, replay_mode);
            let mut src = Source::new(self);
            Self::call_prop_silent(
                |src| {
                    #[cfg(feature = "std")]
                    crate::vprintln!("[chaos_theory] --- determinism check self-replay ---");
                    prop(src);

                    let self_ = src.as_mut();
                    if self_.replay_exact() && self_.rng_used() {
                        self_.signal_replay_mismatch("replay used fresh randomness");
                    }
                    if self_.replay_exact() && !self_.tape_replay.strict_replay_done() {
                        self_.signal_replay_mismatch("replay did not fully consume recorded input");
                    }
                },
                &mut src,
                silent,
            )
        };
        // Restore the original output tape so we report it, and not the one after the replay.
        self.tape_out = tape_out;

        replay_ret.map_err(|mut info| {
            if !info.determinism_failure {
                info.invalid_data = false;
                info.determinism_failure = true;
                info.message = format!(
                    "{DETERMINISM_FAILED_PREFIX}self-replay failed after a successful first pass: {}",
                    info.message
                );
            }
            info
        })
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
        check_determinism: bool,
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
            replay_mode: ReplayMode::Off,
            tape_replay: tape.unwrap_or_default(),
            tape_out: Tape::new(true),
            repeat_noop_ixs: Vec::new(),
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
                check_determinism,
                first_generation: true,
                tape_replay_inactive: Vec::default(),
                mut_cache: MutationCache::default(),
                #[cfg(feature = "std")]
                check_watchdog: None,
            }),
        };
        env.tape_out.reserve_for_replay(&env.tape_replay);
        env
    }

    fn start_from_seed(&mut self, seed: u32, log_depth: usize) -> Source<'_> {
        self.start_from_tape(seed, Tape::default(), log_depth)
    }

    fn init_from_tape(&mut self, seed: u32, tape: Tape, log_depth: usize, replay_mode: ReplayMode) {
        // Invariant is that the tape is valid here, but we don't check it to make it possible
        // to debug us producing invalid tapes (or, more commonly, tape validation errors).
        debug_assert!(tape.reuse_at_zero());
        self.seed = seed;
        self.rng = Rand::new(u64::from(seed));
        self.replay_mode = replay_mode;
        self.tape_replay = tape;
        self.log_depth = log_depth;
        self.budget_remaining = self.slow.budget;
        self.tape_out.clear();
        self.repeat_noop_ixs.clear();
        self.scope_id = ScopeId::default();
        self.scope_depth = 0;
        self.scope_depth_manual = 0;
        self.scope_child_ix = 0;
        self.scope_version = 0;
        self.scope_enum_mode = false;
    }

    fn start_from_tape(&mut self, seed: u32, tape: Tape, log_depth: usize) -> Source<'_> {
        self.init_from_tape(seed, tape, log_depth, ReplayMode::Off);
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

    pub(crate) fn last_event_ix(&self) -> usize {
        self.tape_out.last_event_ix()
    }

    pub(crate) fn repeat_noop_base(&self) -> usize {
        self.repeat_noop_ixs.len()
    }

    pub(crate) fn repeat_noop_push(&mut self, ix: usize) {
        debug_assert!(u32::try_from(ix).is_ok());
        self.repeat_noop_ixs.push(ix as u32);
    }

    pub(crate) fn repeat_noop_mark_discardable(&mut self, from: usize) {
        for &ix in &self.repeat_noop_ixs[from..] {
            self.tape_out.mark_repeat_noop_discardable(ix as usize);
        }
    }

    pub(crate) fn repeat_noop_truncate(&mut self, len: usize) {
        self.repeat_noop_ixs.truncate(len);
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
        log_value_impl(self, label, v);
    }

    fn log_return(&self, v: &impl Debug) {
        if self.log_verbose && self.should_log() {
            self.log_return_impl(v);
        }
    }

    #[inline(never)]
    fn log_return_impl(&self, v: &impl Debug) {
        log_return_impl(self, v);
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
    fn log_scope_enter_impl(
        &self,
        label: &str,
        variant: &str,
        variant_semantic: bool,
        variant_index: usize,
        counter: Option<u32>,
    ) {
        log_scope_enter_impl(
            self,
            label,
            variant,
            variant_semantic,
            variant_index,
            counter,
        );
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
        log_effect_impl(self, label, counter, effect);
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

    fn shadow_replay_mode(&self, scrutiny: bool, check_iter: Option<usize>) -> ReplayMode {
        let want = scrutiny
            || check_iter.is_some_and(|i| self.slow.check_determinism || (i + 1).is_power_of_two());
        if !want {
            ReplayMode::Off
        } else if !self.slow.check_determinism {
            ReplayMode::Lax { diverged: false }
        } else {
            ReplayMode::Strict
        }
    }

    fn replay_exact(&self) -> bool {
        matches!(
            self.replay_mode,
            ReplayMode::Strict | ReplayMode::Lax { diverged: false }
        )
    }

    pub(crate) fn signal_replay_mismatch(&mut self, msg: impl core::fmt::Display) {
        match self.replay_mode {
            ReplayMode::Off => unreachable!("internal error: replay mismatch during non-replay"),
            ReplayMode::Strict => panic_determinism(msg),
            ReplayMode::Lax { ref mut diverged } => {
                if !*diverged {
                    *diverged = true;
                    #[cfg(feature = "std")]
                    eprintln!(
                        "[chaos_theory] warning: determinism self-replay diverged ({msg}); enable `CHAOS_THEORY_CHECK_DETERMINISM=true` to fail on replay divergence"
                    );
                }
            }
        }
    }

    pub(crate) fn observe(&mut self, label: &str, value: u64) {
        if self.log_verbose {
            self.log_value(label, &value);
        }
        if self.replay_exact() && !self.tape_replay.try_pop_observe_exact(value) {
            self.signal_replay_mismatch(format_args!("observation mismatch for `{label}`"));
        }
        self.tape_out.push_observe(value);
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
        print_input(self);
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
        self.init_from_tape(seed, tape, log_depth, ReplayMode::Off);
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

/// Result of the [`Source::repeat`] or [`SourceEx::repeat`](crate::SourceEx::repeat) step.
///
/// Correct `Effect` values make exploration and minimization much more efficient.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[must_use]
pub enum Effect {
    /// Nothing has happened to the system, step can be discarded without affecting the execution.
    ///
    /// Common example is selected action being inapplicable in the current system state.
    Noop,
    /// System state may have changed, but no real progress has been made.
    Change,
    /// Useful work was done.
    Success,
}

impl Arbitrary for Effect {
    fn arbitrary() -> impl Generator<Item = Self> {
        from_next(|src, example| {
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
    fn pop_choice(&mut self, expected: &Event) -> Option<u64> {
        if self.replay_exact() {
            let Some(reuse) = self.tape_replay.try_pop_choice_exact(expected) else {
                self.signal_replay_mismatch("structural mismatch (choice)");
                return self.tape_replay.pop_choice(&mut self.budget_remaining);
            };
            Some(reuse)
        } else {
            self.tape_replay.pop_choice(&mut self.budget_remaining)
        }
    }

    fn choice_new_size(&mut self, n: usize, depth: usize) -> usize {
        // For recursive data, gradually tighten the size distribution.
        // This is a hacky way to work around recursive data generation for cases
        // where recursion is expressed in terms of size (and not e.g. tree-as-enum).
        let t = self.temperature >> (depth / 2);
        let q = mul_add(depth as f64, 0.25, 2.0);
        let d = Biased::new_temperature(t, Some(q));
        d.sample(&mut self.rng, n)
    }

    pub(crate) fn choose_size(&mut self, r: SizeRange, example: Option<usize>) -> usize {
        let expected = Event::Size {
            size: 0,
            min: r.min as u64,
            max: r.max as u64,
        };
        let reuse_extra = self.pop_choice(&expected).map(|u| u as usize);
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
            .unwrap_or_else(|| self.choice_new_size(n, self.scope_depth));
        debug_assert!(size_extra < n);
        let size = r.min + size_extra;
        debug_assert!(r.contains(&size));
        self.budget_remaining = self.budget_remaining.saturating_sub(size.max(1));
        self.tape_out.push_size(size, r);
        size
    }

    fn choice_new_swarm(&mut self, n: u64, tweak: Tweak) -> u64 {
        let tweaked_scope_id = self.scope_id.0.wrapping_add(tweak as u64);
        if self.scope_enum_mode {
            // Get deterministic random value that depends on seed and scope.
            let r = Wyrand::mix(u64::from(self.seed), tweaked_scope_id);
            // Enumerate 2^32 possible choices (when the version is incremented).
            let ix = u64::from(self.scope_version) % n;
            permute(ix, n, r)
        } else {
            // Get deterministic random value that depends on seed, scope and version.
            let r = Wyrand::mix(
                (u64::from(self.seed) << 32) | u64::from(self.scope_version),
                tweaked_scope_id,
            );
            // Use deterministic random value to transform the bound.
            // By using a bound of 2*n instead of n, we only use swarm testing 50% of the time (for small n).
            let m = fast_reduce(r, n.saturating_mul(2));
            if m < n {
                // Do the swarm testing magic.
                permute(self.rng.next_below_u64(m + 1), n, r)
            } else {
                // Just a random choice.
                self.rng.next_below_u64(n)
            }
        }
    }

    pub(crate) fn choose_index(&mut self, n: usize, example: Option<usize>, tweak: Tweak) -> usize {
        debug_assert_ne!(n, 0);
        let forced = self.tape_out.next_choice_forced();
        let expected = Event::Index {
            index: 0,
            max: (n - 1) as u64,
            forced,
        };
        let reuse = self.pop_choice(&expected).map(|u| u as usize);
        debug_assert!(!forced || example.is_some_and(|u| u < n));
        // When out of budget, always choose zero unless the index is structurally forced.
        let n_lim = if self.budget_remaining > 0 || forced {
            n
        } else {
            1
        };
        let example = example.filter(|u| *u < n_lim);
        let index = example
            .or_else(|| reuse.filter(|u| *u < n_lim))
            .unwrap_or_else(|| self.choice_new_swarm(n_lim as u64, tweak) as usize);
        debug_assert!(index < n_lim);
        self.budget_remaining = self.budget_remaining.saturating_sub(n.bit_len().max(1));
        self.tape_out.push_index(index, n);
        index
    }

    pub(crate) fn mark_next_choice_forced(&mut self) {
        self.tape_out.mark_next_choice_forced();
    }

    fn choice_new_value(&mut self, max: u64, bias_to_small: bool) -> u64 {
        if bias_to_small {
            let total_bits = max.bit_len();
            let use_bits = self.size_dist.sample(&mut self.rng, total_bits + 1);
            let mut w = self.rng.next();
            w &= bitmask::<u64>(use_bits);
            w = w.min(max);
            w
        } else {
            self.rng.next_below_u64(max.saturating_add(1))
        }
    }

    pub(crate) fn choose_value(
        &mut self,
        r: Range<u64>,
        example: Option<u64>,
        bias_to_small: bool,
    ) -> u64 {
        // Try to preserve the value bit pattern while ensuring that we fit into required range.
        // This is consistent with our "number-is-a-bit-string" idea.
        fn fit_to_extra(u: Option<u64>, r: Range<u64>, max: u64) -> Option<u64> {
            u.and_then(|u| {
                let max_value = r.min + max;
                if u >= r.min && u <= max_value {
                    return Some(u - r.min);
                }
                let mut v = u & bitmask::<u64>(max_value.bit_len());
                while v > max_value {
                    // Clear the top bit.
                    v &= !(1 << (v.bit_len() - 1));
                }
                if v >= r.min { Some(v - r.min) } else { None }
            })
        }

        let expected = Event::Value {
            value: 0,
            min: r.min,
            max: r.max,
        };
        let reuse_extra = self.pop_choice(&expected);
        // When out of budget, always choose lower bound.
        let max = if self.budget_remaining > 0 {
            r.max - r.min
        } else {
            0
        };
        let example_extra = fit_to_extra(example, r, max);
        let value_extra = example_extra
            .or_else(|| fit_to_extra(reuse_extra, Range::new_raw(0, max), max))
            .unwrap_or_else(|| self.choice_new_value(max, bias_to_small));
        debug_assert!(value_extra <= max);
        let value = r.min + value_extra;
        debug_assert!(r.contains(&value));
        self.budget_remaining = self.budget_remaining.saturating_sub(value.bit_len().max(1));
        self.tape_out.push_value(value, r);
        value
    }

    pub(crate) fn choose_token(&mut self, example: Option<u64>) -> u64 {
        let expected = Event::Token { value: 0 };
        let reuse = self.pop_choice(&expected);
        let value = example.or(reuse).unwrap_or_else(|| self.rng.next());
        self.budget_remaining = self.budget_remaining.saturating_sub(value.bit_len().max(1));
        self.tape_out.push_token(value);
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
        let seed_ix = self.choice_new_swarm(seeds as u64, Tweak::SeedChoice);
        Some(seed_ix as usize)
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
                let _ = g.next(src.as_ex(), seed);
            },
        )
    }

    pub(crate) fn produce_tape(
        seed: u32,
        temperature: u8,
        budget: usize,
        prop: impl Fn(&mut Source),
    ) -> Option<Tape> {
        let mut env = Self::builder()
            .with_rng_seed(seed)
            .with_rng_temperature(temperature)
            .with_rng_budget(budget)
            .build();
        let mut src = env.start_from_seed(seed, 0);
        // TODO: use a version of `filter` here that rolls several times to try to get valid tape?
        let r = Self::call_prop_silent(prop, &mut src, true);
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
        if env.replay_exact() {
            if !env.tape_replay.try_pop_scope_enter_exact(kind, scope_id.0) {
                env.signal_replay_mismatch("structural mismatch (scope enter)");
                env.tape_replay.pop_scope_enter(kind);
            }
        } else {
            env.tape_replay.pop_scope_enter(kind);
        }
        let _ = env.tape_out.push_scope_enter(scope_id.0, kind);
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
        if env.replay_exact() {
            if !env.tape_replay.try_pop_scope_exit_exact(self.effect) {
                #[cfg(feature = "std")]
                let can_report =
                    !(env.replay_mode == ReplayMode::Strict && std::thread::panicking());
                #[cfg(not(feature = "std"))]
                let can_report = true;
                if can_report {
                    env.signal_replay_mismatch("structural mismatch (scope exit)");
                    env.tape_replay.pop_scope_exit();
                }
            }
        } else {
            env.tape_replay.pop_scope_exit();
        }
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
    pub(crate) src: &'source mut SourceEx<'env>,
    should_pop: bool,
}

impl<'source, 'env> SeedTapeReplayScope<'source, 'env> {
    pub(crate) fn new<G: Generator>(
        src: &'source mut SourceEx<'env>,
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
    use crate::{
        assume, check, config::CHECK_ITERS_DEFAULT, make, tests::RgbState, vdbg, vprintln,
    };

    #[test]
    fn forced_index_ignores_exhausted_budget() {
        let mut env = Env::builder().with_rng_budget(0).build();
        env.mark_next_choice_forced();
        assert_eq!(env.choose_index(2, Some(1), Tweak::None), 1);
    }

    #[test]
    fn check_replay_e2e() {
        check(|src| {
            let mut e = Env::builder()
                .with_rng_seed(src.any("seed"))
                .with_rng_temperature(src.any("temperature"))
                .with_rng_budget(src.any("budget"))
                // At least 1 iteration is required, because replay makes one.
                .with_check_iters(src.any_of("check_iters", make::int_in(1..=CHECK_ITERS_DEFAULT)))
                // Limit the time spent reducing to speed the test up.
                .with_reduce_time(Duration::from_millis(50))
                .build();
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

            vprintln!("replaying...");
            let mut f = Env::builder()
                .with_rng_seed(src.any("replay seed"))
                .with_rng_temperature(src.any("replay temperature"))
                .with_rng_budget(e.slow.budget)
                .with_rng_choices(e_result.tape.as_choices().to_vec())
                .with_check_iters(1)
                .with_reduce_time(Duration::ZERO)
                .build();
            let mut f_state = RgbState::default();
            let f_result = f.check_silent(|src| {
                f_state = RgbState::default();
                f_state.prop_fill(src);
            });

            vdbg!((e_result.valid, e_result.invalid, e_result.time_exit));
            vdbg!((f_result.valid, f_result.invalid, f_result.time_exit));
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
            Env::builder().with_check_iters(1).build().check(|_s| {
                i.fetch_add(black_box(1), Ordering::SeqCst);
            });
            i
        });
    }

    #[bench]
    fn choice_new_swarm(b: &mut test::Bencher) {
        let mut env = Env::builder().build();
        b.iter(|| env.choice_new_swarm(black_box(10), black_box(Tweak::None)));
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
        let mut env = Env::builder().with_rng_budget(usize::MAX).build();
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
            let s: Arc<str> = env.generate();
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
            let s: String = env.generate();
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
