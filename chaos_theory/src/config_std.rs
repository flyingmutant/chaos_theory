// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::string::String;
use core::{
    fmt::{Debug, Display},
    time::Duration,
};
use std::sync::{Once, OnceLock};

use crate::{TEMPERATURE_DEFAULT, tape::Tape};

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

#[cfg(test)]
pub(super) fn slow_test_enabled() -> bool {
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
        super::REPLAY_TYPE_EVENTS
    } else {
        super::REPLAY_TYPE_CHOICES
    };
    let replay = super::replay_format(typ, seed, temperature, budget, tape);
    let suffix = if min { " and minimize" } else { "" };
    eprintln!(
        "[chaos_theory] run test with `{REPLAY_VAR}={replay}` environment variable to reproduce{suffix} the failure"
    );
}

pub(super) fn warn_unknown_env_vars(use_env_vars: bool) {
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
}

pub(super) fn replay_fallback(use_ev: bool) -> Option<(u32, u8, usize, Tape)> {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(REPLAY_VAR, None, use_ev.then_some(&ENV), |s| {
        Ok::<_, &'static str>(Some(super::replay_parse(s, true, false)?))
    })
}

pub(super) fn replay_verbose_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        REPLAY_VERBOSE_VAR,
        super::REPLAY_VERBOSE_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

pub(super) fn cover_depth_fallback(use_ev: bool) -> usize {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        COVER_DEPTH_VAR,
        super::COVER_DEPTH_DEFAULT,
        use_ev.then_some(&ENV),
        str::parse,
    )
}

pub(super) fn cover_require_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        COVER_REQUIRE_VAR,
        super::COVER_REQUIRE_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

pub(super) fn check_iters_fallback(use_ev: bool) -> usize {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        CHECK_ITERS_VAR,
        super::CHECK_ITERS_DEFAULT,
        use_ev.then_some(&ENV),
        |s| s.chars().filter(|c| *c != '_').collect::<String>().parse(),
    )
}

pub(super) fn check_time_fallback(use_ev: bool) -> Duration {
    static ENV: OnceLock<String> = OnceLock::new();
    let secs: u64 = param_fallback(
        CHECK_TIME_VAR,
        super::CHECK_TIME_DEFAULT.as_secs(),
        use_ev.then_some(&ENV),
        |s| s.strip_suffix("s").unwrap_or(s).parse(),
    );
    Duration::from_secs(secs)
}

pub(super) fn reduce_time_fallback(use_ev: bool) -> Duration {
    static ENV: OnceLock<String> = OnceLock::new();
    let secs: u64 = param_fallback(
        REDUCE_TIME_VAR,
        super::REDUCE_TIME_DEFAULT.as_secs(),
        use_ev.then_some(&ENV),
        |s| s.strip_suffix("s").unwrap_or(s).parse(),
    );
    Duration::from_secs(secs)
}

pub(super) fn pretty_print_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        PRETTY_PRINT_VAR,
        super::PRETTY_PRINT_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

pub(super) fn log_depth_fallback(use_ev: bool) -> usize {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        LOG_DEPTH_VAR,
        super::LOG_DEPTH_DEFAULT,
        use_ev.then_some(&ENV),
        str::parse,
    )
}

pub(super) fn log_always_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        LOG_ALWAYS_VAR,
        super::LOG_ALWAYS_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

pub(super) fn log_verbose_fallback(use_ev: bool) -> bool {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        LOG_VERBOSE_VAR,
        super::LOG_VERBOSE_DEFAULT,
        use_ev.then_some(&ENV),
        parse_bool,
    )
}

pub(super) fn rng_seed_fallback(use_ev: bool) -> Option<u32> {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(RNG_SEED_VAR, None, use_ev.then_some(&ENV), |s| {
        u32::from_str_radix(s, 16).map(Some)
    })
}

pub(super) fn rng_temperature_fallback(use_ev: bool) -> u8 {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        RNG_TEMPERATURE_VAR,
        TEMPERATURE_DEFAULT,
        use_ev.then_some(&ENV),
        |s| u8::from_str_radix(s, 16),
    )
}

pub(super) fn rng_budget_fallback(use_ev: bool) -> usize {
    static ENV: OnceLock<String> = OnceLock::new();
    param_fallback(
        RNG_BUDGET_VAR,
        super::BUDGET_DEFAULT,
        use_ev.then_some(&ENV),
        |s| usize::from_str_radix(s, 16),
    )
}

pub(super) fn rng_tape_fallback(use_ev: bool) -> Option<Tape> {
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
