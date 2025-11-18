# Design brief

## Baseline Context

- Current core is Env orchestrating runs with seed/budget/temperature plus replay/mutation/reduction and optional coverage.
- Source/SourceRaw provide scoped random-choice APIs (any, choose, repeat, select, maybe, find) and log structured events.
- Generator trait and make::* helpers compose generation; example-guided generation is supported.
- Tape is the trace: choice stream + event stream, with scope metadata and tree-based reduction/mutation.

## Goals

- Language-agnostic, dependency-free core.
- Coherent frontend surface that can power property testing, fuzzing, observability, structural data generation, and composable specs.
- “Smart PRNG” mental model; forward-mode generation; unify Source/SourceRaw; examples optional.
- Faithful representation of control flow and data flow including recursion and concurrency.
- Explicit state-space exploration, coverage, feedback, and “interesting trace” signaling.
- Compose/decompose symmetry: generation from trace and example-to-trace.
- Lifetime-free API; tolerate sloppy traces; allow solver/oracle integration.

## Core Data Structures

- Trace/Tape: ordered event stream with optional metadata and a stable, serializable format.
- Event variants: scope begin/end, choice values, repetition/selection, rejection, control-flow markers, interned labels.
- Scope/path identifiers: stable IDs for choose/select, recursion coordinates, depth counters.
- Choice representation: bit-level and integer-level; stable mapping from values to choices.
- Example linkage: optional example payloads and example-to-trace encoding.
- Coverage/feedback state: hit/miss tracking, custom feedback, interesting-value markers.
- Budget/temperature/state controls: size stack, penalties (new entity/reset), effect/hints.
- Mutation/reduction structures: tree view of trace, mutation caches, crossover inputs.

## Abstractions

- Core PRNG/Source API: choose, repeat, select, maybe/structural-if, rejection sampling.
- Effect (or equivalent) for outcomes and hints.
- Helpers vs. generators: convenient data helpers without centering generator authoring.
- Example-based generation hooks (optional).
- Coverage/instrumentation hooks; trace access/logging.
- Fuzzing hooks: mutation/crossover entrypoints, seed handling.
- Ergonomic utilities: likely!/unlikely!, fail-style hooks, stable key/hash for state signatures.

## Implementation Plan (Skeleton)

- Specify trace/event model, stable IDs, and binary serialization.
- Implement minimal core runtime: seed RNG, scopes, choice ops, trace capture/replay.
- Build compose/decompose: example-to-trace, trace validation.
- Implement minimization and mutation on trace (tree reduction, crossover).
- Add coverage/feedback interfaces and state hashing; integrate hint/penalty mechanics.
- Build frontends: property testing runner, fuzzing harness, observability hooks.
- Add ergonomics and helpers; docs/examples; language bindings.
