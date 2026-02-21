# FAQ

## What is chaos_theory?

chaos_theory is a property‑based testing and structure‑aware fuzzing library.

You explore a system's behavior using structured pseudo-random values and control flow,
chaos_theory ensures that this exploration is efficient and any failures found
are automatically minimized.

## Why did you build it?

I am a huge believer in property-based testing and fuzzing. Having used Go heavily
in the past, I built a property-based library for Go that I quite like:
[rapid](https://github.com/flyingmutant/rapid). Migrating to Rust as my primary language,
I couldn't find something that was both simple and powerful – and I also had
quite a few design ideas left over after implementing rapid. Thus chaos_theory was born.

## How is it different from proptest?

- chaos_theory centers on an imperative `Source` API (`any`, `select`, `repeat`)
  that is suitable for any kind of test, including state machine tests;
  proptest is macro-heavy and has a separate crate for state machine testing support.
- chaos_theory has automatic swarm testing built-in.
- chaos_theory supports example-guided (seeded) data generation.
- chaos_theory uses the same property code for property testing and fuzzing;
  proptest is only for property testing.
- proptest is more mature and feature‑rich; chaos_theory is newer and currently
  lacks a derive macro and recursion helpers.

## How is it different from `arbitrary` and libFuzzer‑style fuzzing?

- `arbitrary` turns raw bytes into values; chaos_theory generates values
  structurally and records the choices that led there.
- libFuzzer mutates bytes; chaos_theory mutates structure, which tends to
  make exploration more efficient by preserving more invariants during mutation.
- In chaos_theory, the same property function is used for property testing and
  fuzzing; the backend changes, not the test.
- chaos_theory still uses libFuzzer (via `libfuzzer_sys`) for fuzzing
  but supplies a structure‑aware mutator and replay format.

If you already have a byte‑based protocol and `arbitrary` derives, that can be
simpler. chaos_theory shines when structure matters and you want unified testing
and fuzzing.

## Do I need to write custom generators?

Usually no. Built‑ins in `make::*` plus `Source::any`, `select`, and `repeat`
cover most tests. Custom generators are fine for domain types or tight
invariants, but they are not a requirement.

## Why are labels important?

Labels are used in the printed reproduction steps. Good labels make the failure
description readable and often point directly to the bug. They also encode
structural information about execution, which improves structure-aware mutation.

## How do I reproduce a failing case?

Use the replay string that chaos_theory prints:

```bash
CHAOS_THEORY_REPLAY=... cargo test failing_test
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
- `Noop`: nothing happened to the system.

Honest `Effect` values make exploration and minimization much more efficient.

## How do I model state machines or stateful systems?

Use `repeat` + `select` and keep a reference model. A common shape is:

- setup SUT and model,
- `repeat` steps that `select` an action,
- apply to both SUT and model,
- assert invariants.

There is no special API for state machines.

## How do I log only the failing case?

Use `vdbg!`, `vprintln!`, or check `Source::should_log`. Logs are scoped and
only emitted for the failing case by default.

## Why do I only see logs from the minimized failure?

By default, chaos_theory logs only for the failing case, and Rust’s test harness
prints captured output for failed tests. That output corresponds to the final
minimized replay.

If you want to see other iterations, enable `CHAOS_THEORY_LOG_ALWAYS=1`.
If you want logs to stream instead of only showing up at the end, run tests with
`-- --nocapture`.

## How do I inspect a few example values quickly?

Use `Env::example` or `Env::example_of` to generate sample values without
running `check`. This is useful for understanding generators and distributions.

## Do I need to worry about distributions or biasing?

Usually no. chaos_theory aims to give a practical default distribution out of the box
with no need for additional tuning: numeric choices skew small, built‑in generators
include curated seed values, boundary values are prioritized, and swarm testing
varies choice subspaces across runs.

If you do have concrete examples, you can seed generators with them. This is
similar to fuzzing dictionaries, but structural and compositional.

## How do I generate constrained data?

Prefer structured generation, as heavy filtering can lead to slow tests.

- Check if there is a ready-made generator for your case
  (for example, `string_matches` or `int_in_range`).
- Consider if you can build values satisfying the constraints directly,
  using tools like `from_fn`, `repeat`, `select`, or generator combinators.

Sometimes constraints are awkward to encode structurally. In those cases,
`filter_assume` or `assume!` are the right tools. Use them deliberately
and keep rejection rates low.

## Why is my test slow?

Common reasons:

- too much work inside a single test case,
- heavy logging,
- too many test cases rejected due to filtering.

Try reducing per‑case work, decreasing logging, or rewriting constraints as
structure instead of filters. The overhead of chaos_theory itself is very low,
so slow tests are almost always about the property or system under test.

## What are the current limitations?

- No derive macro for `Arbitrary` yet.
- Recursion handling is not done yet.
