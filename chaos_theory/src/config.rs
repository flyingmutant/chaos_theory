// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::{format, string::String, vec::Vec};
use core::time::Duration;

use crate::{Env, rand::random_seed_32, tape::Tape};

#[cfg(feature = "std")]
#[path = "config_std.rs"]
mod std_impl;

const REPLAY_VERBOSE_DEFAULT: bool = false;
const COVER_DEPTH_DEFAULT: usize = LOG_DEPTH_DEFAULT;
const COVER_REQUIRE_DEFAULT: bool = false;
pub(crate) const CHECK_ITERS_DEFAULT: usize = 256;
const CHECK_DETERMINISM_DEFAULT: bool = false;
const CHECK_TIME_DEFAULT: Duration = Duration::from_secs(30); // together with REDUCE_TIME_DEFAULT under default slow test warn timeout of 60s
const REDUCE_TIME_DEFAULT: Duration = Duration::from_secs(25);
const PRETTY_PRINT_DEFAULT: bool = false;
const LOG_DEPTH_DEFAULT: usize = 1;
const LOG_ALWAYS_DEFAULT: bool = false;
const LOG_VERBOSE_DEFAULT: bool = false;
pub(crate) const BUDGET_DEFAULT: usize = 0xffffff; // 2^24 - 1

const DELIMITER: &str = "."; // allows for double-click selection in `reproduce_inform` output
const REPLAY_TYPE_CHOICES: &str = "c";
const REPLAY_TYPE_EVENTS: &str = "e";

#[cfg(all(test, feature = "std"))]
pub(crate) fn slow_test_enabled() -> bool {
    std_impl::slow_test_enabled()
}

#[cfg(feature = "std")]
pub(crate) use std_impl::reproduce_inform;

fn replay_format(typ: &str, seed: u32, temperature: u8, budget: usize, tape: &Tape) -> String {
    let t = match typ {
        REPLAY_TYPE_CHOICES => tape.save_choices_base64(),
        REPLAY_TYPE_EVENTS => tape.save_events_base64(),
        _ => unreachable!("internal error: invalid replay type"),
    };
    format!("{typ}{DELIMITER}{seed:x}{DELIMITER}{temperature:x}{DELIMITER}{budget:x}{DELIMITER}{t}")
}

fn replay_parse(
    s: &str,
    validate: bool,
    build_meta: bool,
) -> Result<(u32, u8, usize, Tape), &'static str> {
    let mut it = s.split(DELIMITER);
    let typ = it.next().ok_or("failed to split type")?;
    let seed = it.next().ok_or("failed to split seed")?;
    let seed = u32::from_str_radix(seed, 16).map_err(|_| "failed to parse seed")?;
    let t = it.next().ok_or("failed to split temperature")?;
    let t = u8::from_str_radix(t, 16).map_err(|_| "failed to parse temperature")?;
    let budget = it.next().ok_or("failed to split budget")?;
    let budget = usize::from_str_radix(budget, 16).map_err(|_| "failed to parse budget")?;
    let tape_data = it.next().ok_or("failed to split tape")?;
    let mut tape = Tape::new(build_meta);
    match typ {
        REPLAY_TYPE_CHOICES => {
            tape.load_choices_base64(tape_data.as_bytes())?;
        }
        REPLAY_TYPE_EVENTS => {
            tape.load_events_base64(tape_data.as_bytes(), validate, false)?;
        }
        _ => return Err("invalid replay type"),
    }
    Ok((seed, t, budget, tape))
}

/// Custom configuration for [`Env`].
///
/// Create new config instance with [`Env::custom`].
#[must_use]
#[derive(Debug, Default)]
pub struct Config {
    replay_verbose: Option<bool>,
    cover_depth: Option<usize>,
    cover_require: Option<bool>,
    check_iters: Option<usize>,
    check_determinism: Option<bool>,
    check_time: Option<Duration>,
    reduce_time: Option<Duration>,
    pretty_print: Option<bool>,
    log_depth: Option<usize>,
    log_always: Option<bool>,
    log_verbose: Option<bool>,
    seed: Option<u32>,
    temperature: Option<u8>,
    budget: Option<usize>,
    tape: Option<Tape>,
}

impl Config {
    /// Override replay data (random seed, temperature, budget and choices).
    ///
    /// `replay` should be in format used by `CHAOS_THEORY_REPLAY` environment variable.
    ///
    /// # Errors
    ///
    /// `with_replay` fails when the replay data can't be parsed.
    pub fn with_replay(mut self, replay: &str) -> Result<Self, &'static str> {
        let (seed, temperature, budget, tape) = replay_parse(replay, true, false)?;
        self.seed = Some(seed);
        self.temperature = Some(temperature);
        self.budget = Some(budget);
        self.tape = Some(tape);
        Ok(self)
    }

    /// Override replay (`CHAOS_THEORY_REPLAY`) value verbosity.
    pub fn with_replay_verbose(mut self, enabled: bool) -> Self {
        self.replay_verbose = Some(enabled);
        self
    }

    /// Override coverage collection depth.
    #[doc(hidden)]
    pub fn with_cover_depth(mut self, depth: usize) -> Self {
        self.cover_depth = Some(depth);
        self
    }

    /// Override use of coverage as a testing goal.
    #[doc(hidden)]
    pub fn with_cover_require(mut self, require: bool) -> Self {
        self.cover_require = Some(require);
        self
    }

    /// Override the number of [`Env::check`] iterations.
    pub fn with_check_iters(mut self, check_iters: usize) -> Self {
        self.check_iters = Some(check_iters);
        self
    }

    /// Override whether [`Env::check`] enforces determinism by strict self-replay.
    pub fn with_check_determinism(mut self, enabled: bool) -> Self {
        self.check_determinism = Some(enabled);
        self
    }

    /// Override the time limit for [`Env::check`] (not including test case reduction).
    pub fn with_check_time(mut self, check_time: Duration) -> Self {
        self.check_time = Some(check_time);
        self
    }

    /// Override the time limit for test case reduction.
    pub fn with_reduce_time(mut self, reduce_time: Duration) -> Self {
        self.reduce_time = Some(reduce_time);
        self
    }

    /// Override log pretty-printing.
    pub fn with_pretty_print(mut self, enabled: bool) -> Self {
        self.pretty_print = Some(enabled);
        self
    }

    /// Override log depth.
    pub fn with_log_depth(mut self, depth: usize) -> Self {
        self.log_depth = Some(depth);
        self
    }

    /// Override log being always enabled (instead of only being enabled for minimized failing test case, if any).
    pub fn with_log_always(mut self, enabled: bool) -> Self {
        self.log_always = Some(enabled);
        self
    }

    /// Override log verbosity.
    pub fn with_log_verbose(mut self, enabled: bool) -> Self {
        self.log_verbose = Some(enabled);
        self
    }

    /// Override starting random seed.
    pub fn with_rng_seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Override random temperature.
    pub fn with_rng_temperature(mut self, temperature: u8) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Override random budget.
    pub fn with_rng_budget(mut self, budget: usize) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Specify pseudo-random choices upfront.
    pub fn with_rng_choices(self, choices: Vec<u64>) -> Self {
        self.with_rng_tape(Tape::from_choices(choices))
    }

    pub(crate) fn with_rng_tape(mut self, tape: Tape) -> Self {
        self.tape = Some(tape);
        self
    }

    /// Construct an [`Env`] with this config.
    ///
    /// To determine the `Env` parameters:
    /// - any values specified explicitly by `Config` method calls are used as-is,
    /// - otherwise, in `std`, if `use_env_vars` is true, replay data (seed, temperature, budget and
    ///   choices) encoded in `CHAOS_THEORY_REPLAY` environment variable is used,
    /// - otherwise, in `std`, if `use_env_vars` is true, values from the following environment
    ///   variables are used:
    ///   - `CHAOS_THEORY_COVER_DEPTH`,
    ///   - `CHAOS_THEORY_COVER_REQUIRE`,
    ///   - `CHAOS_THEORY_CHECK_ITERS`,
    ///   - `CHAOS_THEORY_CHECK_DETERMINISM`,
    ///   - `CHAOS_THEORY_CHECK_TIME`,
    ///   - `CHAOS_THEORY_REDUCE_TIME`,
    ///   - `CHAOS_THEORY_PRETTY_PRINT`,
    ///   - `CHAOS_THEORY_LOG_DEPTH`,
    ///   - `CHAOS_THEORY_LOG_ALWAYS`,
    ///   - `CHAOS_THEORY_LOG_VERBOSE`,
    ///   - `CHAOS_THEORY_RNG_SEED`,
    ///   - `CHAOS_THEORY_RNG_TEMPERATURE`,
    ///   - `CHAOS_THEORY_RNG_BUDGET`,
    ///   - `CHAOS_THEORY_RNG_CHOICES`,
    ///   - `CHAOS_THEORY_REPLAY_VERBOSE`,
    /// - otherwise, default parameter values are used.
    ///
    /// In `no_std`, `use_env_vars` is ignored and only explicit values plus defaults are used.
    ///
    /// The random seed defaults to a platform-random value in `std`, and to the deterministic
    /// `no_std` seed sequence otherwise.
    #[must_use]
    pub fn env(self, use_env_vars: bool) -> Env {
        #[cfg(feature = "std")]
        {
            let mut config = self;
            std_impl::warn_unknown_env_vars(use_env_vars);

            if let Some((seed, temperature, budget, tape)) = std_impl::replay_fallback(use_env_vars)
            {
                config.seed = config.seed.or(Some(seed));
                config.temperature = config.temperature.or(Some(temperature));
                config.budget = config.budget.or(Some(budget));
                config.tape = config.tape.or(Some(tape));
            }

            let seed = config.seed.unwrap_or_else(|| {
                std_impl::rng_seed_fallback(use_env_vars).unwrap_or_else(random_seed_32)
            });
            let temperature = config
                .temperature
                .unwrap_or_else(|| std_impl::rng_temperature_fallback(use_env_vars));
            let budget = config
                .budget
                .unwrap_or_else(|| std_impl::rng_budget_fallback(use_env_vars));
            let tape = config
                .tape
                .or_else(|| std_impl::rng_tape_fallback(use_env_vars));

            let cover_depth = config
                .cover_depth
                .unwrap_or_else(|| std_impl::cover_depth_fallback(use_env_vars));
            let cover_require = config
                .cover_require
                .unwrap_or_else(|| std_impl::cover_require_fallback(use_env_vars));
            let check_iters = config
                .check_iters
                .unwrap_or_else(|| std_impl::check_iters_fallback(use_env_vars));
            let check_determinism = config
                .check_determinism
                .unwrap_or_else(|| std_impl::check_determinism_fallback(use_env_vars));
            let check_time = config
                .check_time
                .unwrap_or_else(|| std_impl::check_time_fallback(use_env_vars));
            let reduce_time = config
                .reduce_time
                .unwrap_or_else(|| std_impl::reduce_time_fallback(use_env_vars));
            let pretty_print = config
                .pretty_print
                .unwrap_or_else(|| std_impl::pretty_print_fallback(use_env_vars));
            let log_depth = config
                .log_depth
                .unwrap_or_else(|| std_impl::log_depth_fallback(use_env_vars));
            let log_always = config
                .log_always
                .unwrap_or_else(|| std_impl::log_always_fallback(use_env_vars));
            let log_verbose = config
                .log_verbose
                .unwrap_or_else(|| std_impl::log_verbose_fallback(use_env_vars));
            let replay_verbose = config
                .replay_verbose
                .unwrap_or_else(|| std_impl::replay_verbose_fallback(use_env_vars));

            Env::with_params(
                seed,
                temperature,
                budget,
                tape,
                cover_depth,
                cover_require,
                check_iters,
                check_determinism,
                check_time,
                reduce_time,
                pretty_print,
                log_depth,
                log_always,
                log_verbose,
                replay_verbose,
            )
        }

        #[cfg(not(feature = "std"))]
        {
            let _ = use_env_vars;
            Env::with_params(
                self.seed.unwrap_or_else(random_seed_32),
                self.temperature.unwrap_or(crate::env::TEMPERATURE_DEFAULT),
                self.budget.unwrap_or(BUDGET_DEFAULT),
                self.tape,
                self.cover_depth.unwrap_or(COVER_DEPTH_DEFAULT),
                self.cover_require.unwrap_or(COVER_REQUIRE_DEFAULT),
                self.check_iters.unwrap_or(CHECK_ITERS_DEFAULT),
                self.check_determinism.unwrap_or(CHECK_DETERMINISM_DEFAULT),
                self.check_time.unwrap_or(CHECK_TIME_DEFAULT),
                self.reduce_time.unwrap_or(REDUCE_TIME_DEFAULT),
                self.pretty_print.unwrap_or(PRETTY_PRINT_DEFAULT),
                self.log_depth.unwrap_or(LOG_DEPTH_DEFAULT),
                self.log_always.unwrap_or(LOG_ALWAYS_DEFAULT),
                self.log_verbose.unwrap_or(LOG_VERBOSE_DEFAULT),
                self.replay_verbose.unwrap_or(REPLAY_VERBOSE_DEFAULT),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, tests::RgbState};

    #[test]
    fn replay_parse_format_roundtrip_choices() {
        check(|src| {
            let seed = src.any("seed");
            let temperature = src.any("temperature");
            let budget = src.any("budget");
            let tape = Tape::from_choices(src.any("choices"));
            let s = replay_format(REPLAY_TYPE_CHOICES, seed, temperature, budget, &tape);
            let (seed_, temperature_, budget_, tape_) = replay_parse(&s, false, false).unwrap();
            assert_eq!(seed, seed_);
            assert_eq!(temperature, temperature_);
            assert_eq!(budget, budget_);
            assert_eq!(tape, tape_);
        });
    }

    #[test]
    fn replay_parse_format_roundtrip_events() {
        check(|src| {
            let seed = src.any("seed");
            let temperature = src.any("temperature");
            let budget = src.any("budget");
            let tape = RgbState::default().prop_fill_tape(src, false, false, true);
            let s = replay_format(REPLAY_TYPE_EVENTS, seed, temperature, budget, &tape);
            let (seed_, temperature_, budget_, tape_) =
                replay_parse(&s, false, tape.has_meta()).unwrap();
            assert_eq!(seed, seed_);
            assert_eq!(temperature, temperature_);
            assert_eq!(budget, budget_);
            assert_eq!(tape, tape_);
        });
    }
}
