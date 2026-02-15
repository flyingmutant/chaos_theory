# Config And Repro

This document covers how to control chaos_theory runs and how to reproduce failures. It is intentionally brief.

## The Fast Path: Replay

When a property fails, chaos_theory prints a `CHAOS_THEORY_REPLAY=...` string. Use it to reproduce the case:

```bash
CHAOS_THEORY_REPLAY=... cargo test
```

Replay strings are typically already minimized, so you get the smallest failing case immediately.

You can also set replay programmatically:

```rust
use chaos_theory::Env;

let env = Env::custom().with_replay("...")?.env(true);
```

## Check And Reduction Limits

- `with_check_iters` controls how many valid test cases to try.
- `with_check_time` is a time limit for generation and checking (not reduction).
- `with_reduce_time` is the time limit for test case reduction.

If generation is slow, time limits may stop early. Reduction runs after a failure and has its own limit.

## Logging Controls

Logging is scoped and intentionally quiet by default. The core idea is:

- logs are only shown for the relevant failing case,
- scope depth controls how deep logs are emitted,
- verbosity controls how much detail is printed.

Options:

- `with_log_depth`: how deep logs are emitted (0 disables scoped logs).
- `with_log_always`: always log, even for passing cases.
- `with_log_verbose`: include extra details in logs.
- `with_pretty_print`: pretty-print values in logs.

If you use `vdbg!` or `vprintln!`, they automatically obey these settings.

## RNG Controls (Advanced)

Most users never touch these. They are useful for debugging or exploration experiments.

- `with_rng_seed`: fixed seed for reproducibility.
- `with_rng_temperature`: biases the randomness distribution.
- `with_rng_budget`: limits total randomness consumed.
- `with_rng_choices`: provide explicit choices.

If you hit the RNG budget warning, the test is doing too much work per case or is stuck in too many retries.

## Environment Variables

All config options can be set via environment variables. These are the ones chaos_theory recognizes:

- `CHAOS_THEORY_REPLAY`: replay data (seed, temperature, budget, choices)
- `CHAOS_THEORY_REPLAY_VERBOSE`: include extra replay info
- `CHAOS_THEORY_CHECK_ITERS`: number of test iterations
- `CHAOS_THEORY_CHECK_TIME`: time limit for generation + checking
- `CHAOS_THEORY_REDUCE_TIME`: time limit for reduction
- `CHAOS_THEORY_PRETTY_PRINT`: pretty-print values in logs
- `CHAOS_THEORY_LOG_DEPTH`: log depth
- `CHAOS_THEORY_LOG_ALWAYS`: always emit logs
- `CHAOS_THEORY_LOG_VERBOSE`: extra log detail
- `CHAOS_THEORY_RNG_SEED`: fixed RNG seed
- `CHAOS_THEORY_RNG_TEMPERATURE`: RNG temperature
- `CHAOS_THEORY_RNG_BUDGET`: RNG budget
- `CHAOS_THEORY_RNG_CHOICES`: explicit RNG choices

Coverage-related variables also exist but are not part of the public API surface.
