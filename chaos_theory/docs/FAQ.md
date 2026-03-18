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

## How is it different from `proptest`?

chaos_theory is built around simple macro-free imperative API, has fuzzing support,
and cool tricks like built-in swarm testing and example-based data generation.

`proptest` is more mature and feature‑rich; chaos_theory is newer and currently
lacks some important features like recursion helpers.

## How is it different from `arbitrary` plus `libfuzzer_sys`?

chaos_theory has fuzzing as an optional extra you can use if and when you want to.

chaos_theory gives you much more flexible data generation, more efficient state space
exploration (thanks to structural mutation), and better minimization (again, thanks
to being fully structure-aware). It still uses `libfuzzer_sys` for the main fuzzing
loop and corpus management, but fully takes over data generation and mutation.

If you have a byte‑based protocol and `arbitrary` derives, that can be simpler.
chaos_theory shines when structure matters and you want unified testing and fuzzing.

## How do I model state machines or stateful systems?

Use [`repeat`][source_repeat] + [`select`][source_select] and keep a reference model.
A common shape is:

- setup SUT and model,
- `repeat` steps that `select` an action,
- apply to both SUT and model,
- assert invariants.

There is no special API for state machines.

## Do I need to worry about distributions or biasing?

Usually no. chaos_theory aims to give a practical default distribution out of the box
with no need for additional tuning: choices skew small to prefer interesting subspaces,
built‑in generators include curated seed values, boundary values are prioritized,
and swarm testing varies choice subspaces across runs.

If you do have concrete examples, you can seed generators with them. This is
similar to fuzzing dictionaries, but structural and compositional.

## How do I generate constrained data?

Prefer structured generation, as heavy filtering can lead to slow or failed tests.

- Check if there is a ready-made generator for your case
  (for example, [`string_matching`][make_string_matching] or
  [`int_in_range`][make_int_in_range]).
- Consider if you can build values satisfying the constraints directly,
  using tools like [`from_fn`][make_from_fn], [`repeat`][source_repeat],
  [`select`][source_select], or [`Generator`][generator] combinators.

Sometimes constraints are awkward to encode structurally. In those cases,
[`filter_assume`][generator_filter_assume] or [`assume!`][assume] are
the right tools. Use them deliberately and keep rejection rates low.

## Why did `check` fail with “only generated N valid tests”?

The property rejected too many generated values. Common causes:

- [`assume!`][assume] is used too often.
- [`filter_assume`][generator_filter_assume] predicates reject too much.
- [`repeat`][source_repeat] steps return [`Noop`][effect_noop] too often.

Reduce rejection rates, generate constrained data structurally,
or make invalid cases part of your model instead of discarding them.

## Why did I get a "determinism self-replay diverged" warning?

Your property produced a different execution trace when replayed with its own output.
Common causes: thread-local state, global counters, system time, or other external
dependencies that change between runs.

Set `CHAOS_THEORY_CHECK_DETERMINISM=true` to turn the warning into a hard failure.
Use [`Source::observe`][source_observe] to make internal system state visible to chaos_theory
and help pinpoint determinism-related failures.

## How do I avoid log spam from all successful `check` iterations?

Use [`vdbg!`][vdbg], [`vprintln!`][vprintln], or check
[`Source::should_log`][source_should_log]. chaos_theory logs are scoped and only emitted
for the failing case by default.

If you *do* want to see everything, enable `CHAOS_THEORY_LOG_ALWAYS=1`.

[source]: crate::Source
[source_repeat]: crate::Source::repeat
[source_select]: crate::Source::select
[source_should_log]: crate::Source::should_log

[effect_noop]: crate::Effect::Noop

[generator]: crate::Generator
[generator_filter_assume]: crate::Generator::filter_assume
[generator_seeded]: crate::Generator::seeded

[check]: crate::check
[assume]: crate::assume
[vdbg]: crate::vdbg
[vprintln]: crate::vprintln

[make_from_fn]: crate::make::from_fn
[make_int_in_range]: crate::make::int_in_range
[make_string_matching]: crate::make::string_matching
[source_observe]: crate::Source::observe
