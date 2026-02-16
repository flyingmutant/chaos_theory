# FAQ

## What is chaos_theory?

chaos_theory is a property‑based testing and structure‑aware fuzzing library.
You write properties against structured values, and the engine records the
choices so failures can be reproduced, minimized, and mutated.

## How is it different from proptest?

- chaos_theory centers on an imperative `Source` API (`any`, `select`, `repeat`)
  and treats structure as part of the test logic.
- proptest centers on strategy combinators and a separate shrinker layer.
- chaos_theory uses the same property code for property testing and fuzzing;
  proptest is primarily for property testing.
- In chaos_theory, state machine testing does not use a separate API surface; it
  is just `repeat` + `select` in normal properties.
- chaos_theory’s example mechanism is bidirectional and enables structure‑aware
  mutation and seeded/example‑guided generation.
- proptest is more mature and feature‑rich; chaos_theory is newer and currently
  lacks a derive macro and recursion helpers.

## How is it different from `arbitrary` and libFuzzer‑style fuzzing?

- `arbitrary` turns raw bytes into values; chaos_theory generates values
  structurally and records the choices that led there.
- libFuzzer mutates bytes; chaos_theory mutates structure, which tends to
  preserve shape and invariants.
- In chaos_theory, the same property function is used for property testing and
  fuzzing; the backend changes, not the test.
- chaos_theory still uses libFuzzer (via `libfuzzer_sys`) but supplies a
  structure‑aware mutator and replay format.

If you already have a byte‑based protocol and `arbitrary` derives, that can be
simpler. chaos_theory shines when structure matters and you want unified testing
and fuzzing.

## Do I need to write custom generators?

Usually no. Built‑ins in `make::*` plus `Source::any`, `select`, and `repeat`
cover most tests. Custom generators are fine for domain types or tight
invariants, but they are not a requirement.

## Why are labels important if they’re optional?

Labels are used in replay output and in the printed reproduction steps. Good
labels make the failure description readable and often point directly to the
bug.

## How do I reproduce a failing case?

Use the replay string that chaos_theory prints:

```bash
CHAOS_THEORY_REPLAY=... cargo test
```

Replay strings are typically already minimized.

## Why did `check` stop early with “not enough valid cases”?

The property rejected too many generated values. Common causes:

- `assume!` is used too often.
- `filter_assume` predicates reject too much.
- `repeat` steps return `Noop` too often.

Prefer recoverable `filter`, reduce rejection rates, or make invalid cases part
of your model instead of discarding them.

## What does `Effect` mean in `repeat`?

- `Success`: useful work was done.
- `Change`: state may have changed, but no clear progress.
- `Noop`: nothing happened.

Honest `Effect` values are critical for minimization to work well.

## How do I model state machines or stateful systems?

Use `repeat` + `select` and keep a reference model. A common shape is:

- setup SUT and model,
- `repeat` steps that `select` an action,
- apply to both SUT and model,
- assert invariants.

There is no special API for state machines; the normal API is the advanced mode.

## How do I log only the failing case?

Use `vdbg!`, `vprintln!`, or check `Source::should_log`. Logs are scoped and
only emitted for the failing case by default.

## Why do I only see logs from the minimized failure?

By default, chaos_theory logs only for the failing case, and Rust’s test harness
prints captured output for failed tests. That output corresponds to the final
minimized replay.

If you want to see other iterations, enable `CHAOS_THEORY_LOG_ALWAYS=1` and
increase `CHAOS_THEORY_LOG_DEPTH`. If you want logs to stream instead of only
showing up at the end, run tests with `-- --nocapture`.

## How do I inspect a few example values quickly?

Use `Env::example` or `Env::example_of` to generate sample values without
running `check`. This is useful for understanding generators and distributions.

## Do I need to worry about distributions or biasing?

Usually no. chaos_theory already biases toward edge cases and structurally
interesting values (for example, built‑in special‑case seeds for bytes/strings
and structured float generation).

If you do have concrete examples, you can seed generators with them. This is
similar to fuzzing dictionaries, but structural and compositional.

## How do I generate constrained data?

Prefer structured generation:

- use `select` to choose variants,
- use `repeat` to build sequences,
- use `filter` only when constraints can’t be built in.

If you must use `assume!`, keep it rare.

## Why is my test slow?

Common reasons:

- too much work inside a single test case,
- heavy logging,
- too many rejected cases.

Try reducing per‑case work, lowering log depth, or rewriting constraints as
structure instead of filters. The overhead of chaos_theory itself is
intentionally low, so slow tests are usually about the property or system under
test.

## What does chaos_theory do automatically for me?

- Reproducible failures with minimal cases.
- Structure‑aware mutation for fuzzing.
- Built‑in edge‑case biasing.
- Universal swarm testing baked into choice selection.
- A single property definition for both PBT and fuzzing.

## What are the current limitations?

- No derive macro for `Arbitrary` yet.
- Recursion handling is not done yet.
