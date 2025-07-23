// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    fmt::{Debug, Display},
    time::Duration,
};
use std::sync::{Once, OnceLock};

use crate::{Env, TEMPERATURE_DEFAULT, rand::random_seed_32, tape::Tape};

const VAR_PREFIX: &str = "CHAOS_THEORY_";

const REPLAY_VAR: &str = "CHAOS_THEORY_REPLAY";
const REPLAY_VERBOSE_VAR: &str = "CHAOS_THEORY_REPLAY_VERBOSE";
const COVER_DEPTH_VAR: &str = "CHAOS_THEORY_COVER_DEPTH";
const COVER_REQUIRE_VAR: &str = "CHAOS_THEORY_COVER_REQUIRE";
const CHECK_ITERS_VAR: &str = "CHAOS_THEORY_CHECK_ITERS";
const CHECK_TIME_VAR: &str = "CHAOS_THEORY_CHECK_TIME";
const REDUCE_TIME_VAR: &str = "CHAOS_THEORY_REDUCE_TIME";
const PRETTY_PRINT_VAR: &str = "CHAOS_THEORY_PRETTY_PRINT";
const LOG_DEPTH_VAR: &str = "CHAOS_THEORY_LOG_DEPTH";
const LOG_ALWAYS_VAR: &str = "CHAOS_THEORY_LOG_ALWAYS";
const LOG_VERBOSE_VAR: &str = "CHAOS_THEORY_LOG_VERBOSE";
const RNG_SEED_VAR: &str = "CHAOS_THEORY_RNG_SEED";
const RNG_TEMPERATURE_VAR: &str = "CHAOS_THEORY_RNG_TEMPERATURE";
const RNG_BUDGET_VAR: &str = "CHAOS_THEORY_RNG_BUDGET";
const RNG_CHOICES_VAR: &str = "CHAOS_THEORY_RNG_CHOICES";

// Internal.
const SLOW_TEST_VAR: &str = "CHAOS_THEORY_SLOW_TEST";

const KNOWN_CONFIG_VARS: &[&str] = &[
    REPLAY_VAR,
    REPLAY_VERBOSE_VAR,
    COVER_DEPTH_VAR,
    COVER_REQUIRE_VAR,
    CHECK_ITERS_VAR,
    CHECK_TIME_VAR,
    REDUCE_TIME_VAR,
    PRETTY_PRINT_VAR,
    LOG_DEPTH_VAR,
    LOG_ALWAYS_VAR,
    LOG_VERBOSE_VAR,
    RNG_SEED_VAR,
    RNG_TEMPERATURE_VAR,
    RNG_BUDGET_VAR,
    RNG_CHOICES_VAR,
    SLOW_TEST_VAR,
];

const REPLAY_VERBOSE_DEFAULT: bool = false;
const COVER_DEPTH_DEFAULT: usize = LOG_DEPTH_DEFAULT;
const COVER_REQUIRE_DEFAULT: bool = false;
pub(crate) const CHECK_ITERS_DEFAULT: usize = 256;
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

#[cfg(test)]
pub(crate) fn slow_test_enabled() -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(SLOW_TEST_VAR, false, Some(&ENV), parse_bool)
}

pub(crate) fn reproduce_inform(
    seed: u32,
    temperature: u8,
    budget: usize,
    tape: &Tape,
    verbose: bool,
    min: bool,
) {
    let typ = if verbose {
        // Outputting events from fuzzer runs is more convenient, since fuzzer expects events as input, too.
        REPLAY_TYPE_EVENTS
    } else {
        REPLAY_TYPE_CHOICES
    };
    let r = replay_format(typ, seed, temperature, budget, tape);
    let m = if min { " and minimize" } else { "" };
    eprintln!(
        "[chaos_theory] run test with `{REPLAY_VAR}={r}` environment variable to reproduce{m} the failure"
    );
}

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

// Debug bound on T should really be a Display one.
fn param_fallback<T: Debug, E: Display>(
    name: &'static str,
    default_: T,
    use_ev: Option<&OnceLock<String>>,
    parse: impl FnOnce(&str) -> Result<T, E>,
) -> T {
    let s = if let Some(ev) = use_ev {
        // Maybe it would be a bit better to cache the parsed result?
        // That would at least allow `eprintln` below to be shown once.
        ev.get_or_init(|| std::env::var(name).unwrap_or_default())
    } else {
        ""
    };
    if s.is_empty() {
        return default_;
    }
    let r = parse(s);
    r.unwrap_or_else(|e| {
        eprintln!(
            "[chaos_theory] failed to parse {name} value {s:?}, using default {default_:?} ({e})"
        );
        default_
    })
}

fn parse_bool(s: &str) -> Result<bool, &'static str> {
    match s {
        "true" | "on" | "yes" | "y" | "1" => Ok(true),
        "false" | "off" | "no" | "n" | "0" => Ok(false),
        _ => Err("invalid boolean value"),
    }
}

fn replay_fallback(use_ev: bool) -> Option<(u32, u8, usize, Tape)> {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(REPLAY_VAR, None, use_ev.then_some(&ENV), |s| {
        Ok::<_, &'static str>(Some(replay_parse(s, true, false)?))
    })
}

fn replay_verbose_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        REPLAY_VERBOSE_VAR,
        REPLAY_VERBOSE_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

fn cover_depth_fallback(use_ev: bool) -> usize {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        COVER_DEPTH_VAR,
        COVER_DEPTH_DEFAULT,
        use_ev.then_some(&ENV),
        str::parse,
    )
}

fn cover_require_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        COVER_REQUIRE_VAR,
        COVER_REQUIRE_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

fn check_iters_fallback(use_ev: bool) -> usize {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        CHECK_ITERS_VAR,
        CHECK_ITERS_DEFAULT,
        use_ev.then_some(&ENV),
        |s| {
            let mut s = s.to_owned();
            s.retain(|c| c != '_');
            s.parse()
        },
    )
}

fn check_time_fallback(use_ev: bool) -> Duration {
    static ENV: OnceLock<String> = OnceLock::new();
    let secs: u64 = param_fallback(
        CHECK_TIME_VAR,
        CHECK_TIME_DEFAULT.as_secs(),
        use_ev.then_some(&ENV),
        |s| s.strip_suffix("s").unwrap_or(s).parse(),
    );
    Duration::from_secs(secs)
}

fn reduce_time_fallback(use_ev: bool) -> Duration {
    static ENV: OnceLock<String> = OnceLock::new();
    let secs: u64 = param_fallback(
        REDUCE_TIME_VAR,
        REDUCE_TIME_DEFAULT.as_secs(),
        use_ev.then_some(&ENV),
        |s| s.strip_suffix("s").unwrap_or(s).parse(),
    );
    Duration::from_secs(secs)
}

fn pretty_print_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        PRETTY_PRINT_VAR,
        PRETTY_PRINT_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

fn log_depth_fallback(use_ev: bool) -> usize {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        LOG_DEPTH_VAR,
        LOG_DEPTH_DEFAULT,
        use_ev.then_some(&ENV),
        str::parse,
    )
}

fn log_always_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        LOG_ALWAYS_VAR,
        LOG_ALWAYS_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

fn log_verbose_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        LOG_VERBOSE_VAR,
        LOG_VERBOSE_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

fn rng_seed_fallback(use_ev: bool) -> Option<u32> {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(RNG_SEED_VAR, None, use_ev.then_some(&ENV), |s| {
        u32::from_str_radix(s, 16).map(Some)
    })
}

fn rng_temperature_fallback(use_ev: bool) -> u8 {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        RNG_TEMPERATURE_VAR,
        TEMPERATURE_DEFAULT,
        use_ev.then_some(&ENV),
        |s| u8::from_str_radix(s, 16),
    )
}

fn rng_budget_fallback(use_ev: bool) -> usize {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        RNG_BUDGET_VAR,
        BUDGET_DEFAULT,
        use_ev.then_some(&ENV),
        |s| usize::from_str_radix(s, 16),
    )
}

fn rng_tape_fallback(use_ev: bool) -> Option<Tape> {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(RNG_CHOICES_VAR, None, use_ev.then_some(&ENV), |s| {
        // A bit hacky: because we don't distinguish between empty and unset
        // environment variables, it is not possible to specify an empty tape.
        debug_assert!(!s.is_empty());
        let mut tape = Tape::default();
        tape.load_choices_base64(s.as_bytes())?;
        Ok::<_, &'static str>(Some(tape))
    })
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
    pub fn with_cover_depth(mut self, depth: usize) -> Self {
        self.cover_depth = Some(depth);
        self
    }

    /// Override use of coverage as a testing goal.
    pub fn with_cover_require(mut self, require: bool) -> Self {
        self.cover_require = Some(require);
        self
    }

    /// Override the number of [`Env::check`] iterations.
    pub fn with_check_iters(mut self, check_iters: usize) -> Self {
        self.check_iters = Some(check_iters);
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
    /// - otherwise, if `use_env_vars` is true, replay data (seed, temperature, budget and choices)
    ///   encoded in `CHAOS_THEORY_REPLAY` environment variable is used,
    /// - otherwise, if `use_env_vars` is true, values from the following environment variables are used:
    ///   - `CHAOS_THEORY_COVER_DEPTH`,
    ///   - `CHAOS_THEORY_COVER_REQUIRE`,
    ///   - `CHAOS_THEORY_CHECK_ITERS`,
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
    /// Note that random seed defaults to random value.
    #[must_use]
    pub fn env(mut self, use_env_vars: bool) -> Env {
        static CHECK_ENV_ONCE: Once = Once::new();
        if use_env_vars {
            CHECK_ENV_ONCE.call_once(|| {
                for (var, _) in std::env::vars_os() {
                    if let Some(var) = var.to_str()
                        && var.starts_with(VAR_PREFIX)
                        && !KNOWN_CONFIG_VARS.contains(&var)
                    {
                        eprintln!(
                            "[chaos_theory] warning: unknown environment variable {var}, ignoring"
                        );
                    }
                }
            });
        }

        if let Some((seed, temperature, budget, tape)) = replay_fallback(use_env_vars) {
            self.seed = self.seed.or(Some(seed));
            self.temperature = self.temperature.or(Some(temperature));
            self.budget = self.budget.or(Some(budget));
            self.tape = self.tape.or(Some(tape));
        }

        let seed = self
            .seed
            .unwrap_or_else(|| rng_seed_fallback(use_env_vars).unwrap_or_else(random_seed_32));
        let temperature = self
            .temperature
            .unwrap_or_else(|| rng_temperature_fallback(use_env_vars));
        let budget = self
            .budget
            .unwrap_or_else(|| rng_budget_fallback(use_env_vars));
        let tape = self.tape.or_else(|| rng_tape_fallback(use_env_vars));

        let cover_depth = self
            .cover_depth
            .unwrap_or_else(|| cover_depth_fallback(use_env_vars));
        let cover_require = self
            .cover_require
            .unwrap_or_else(|| cover_require_fallback(use_env_vars));
        let check_iters = self
            .check_iters
            .unwrap_or_else(|| check_iters_fallback(use_env_vars));
        let check_time = self
            .check_time
            .unwrap_or_else(|| check_time_fallback(use_env_vars));
        let reduce_time = self
            .reduce_time
            .unwrap_or_else(|| reduce_time_fallback(use_env_vars));
        let pretty_print = self
            .pretty_print
            .unwrap_or_else(|| pretty_print_fallback(use_env_vars));
        let log_depth = self
            .log_depth
            .unwrap_or_else(|| log_depth_fallback(use_env_vars));
        let log_always = self
            .log_always
            .unwrap_or_else(|| log_always_fallback(use_env_vars));
        let log_verbose = self
            .log_verbose
            .unwrap_or_else(|| log_verbose_fallback(use_env_vars));
        let replay_verbose = self
            .replay_verbose
            .unwrap_or_else(|| replay_verbose_fallback(use_env_vars));

        Env::with_params(
            seed,
            temperature,
            budget,
            tape,
            cover_depth,
            cover_require,
            check_iters,
            check_time,
            reduce_time,
            pretty_print,
            log_depth,
            log_always,
            log_verbose,
            replay_verbose,
        )
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
