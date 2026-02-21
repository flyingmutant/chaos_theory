# Internals

This document explains how chaos_theory works conceptually. It focuses on the
core machinery that makes replay, minimization, and structure-aware fuzzing
possible.

## Mental Model

Think of a test run as an interaction between three pieces:

- `Env` drives the run and owns configuration.
- `Source` produces structured randomness.
- `Tape` records the choices and events so the run can be replayed, minimized, and mutated.

## How `check` Runs

At a high level, `Env::check` does this:

1. Create a fresh `Source` for each iteration.
2. Execute the property and record all choices and events.
3. Count valid vs invalid cases.
4. On failure, replay and then minimize the failing trace.

Invalid cases are those where the property calls `assume!` or a generator
rejects values too often. Too many invalid cases will stop the run early.

## Source, Scopes, and Choices

`Source` is the only place randomness should come from. It provides operations
like `any`, `choose`, `select`, `repeat`, and `maybe`.

Each operation creates a scope with a label and records the choices inside it.
Labels are not required for correctness, but they are used in replay output and
make the failure description readable.

## Generators and Examples

Generators are composable descriptions of how to produce values. Every generator
has a `next` method that takes an optional `example` reference.

The `example` is not a hint. It is the reverse direction of generation: when an
example is provided, the generator should reconstruct the exact choice trace
that would produce that value. This is a reflective, parser/serializer-like
symmetry.

This symmetry is central to chaos_theory. It allows the engine to build a `Tape`
from a concrete value, then mutate that tape to generate structurally similar
values. That is how example-guided generation and seeded dictionaries work for
complex data.

In practice, example handling is mixed with fresh randomness. If a part of the
value can be reconstructed, it is encoded into the tape. If not, the generator
falls back to fresh choices for that part.

`map_reversible` exists to preserve this property for transforms, and custom
generators should always thread `example` through to sub-generators.

A core invariant is that any value produced by a generator is a valid
forward-mode value that obeys all generator invariants by construction,
regardless of whether it came from randomness, examples, seeds, or minimization.

## `select` and `repeat`

`select` chooses a variant by label and records the choice. This is how you
describe structured alternatives and keep them stable across replay.

`repeat` builds sequences of steps and relies on `Effect` to understand what happened:

- `Success` means useful work was done.
- `Change` means the state may have changed, but there was no clear progress.
- `Noop` means nothing happened.

## The Trace (Choices and Events)

Internally, `Tape` stores two streams:

- A compact stream of raw choice values.
- A richer stream of events that includes scope boundaries and metadata.

Together, they form a trace that can be replayed deterministically. The raw
choice stream exists to keep replay strings compact. `CHAOS_THEORY_REPLAY=...`
encodes that stream by default.

## Replay

Replay runs the property using the recorded choices instead of new randomness.
This guarantees that the same structure and values are reconstructed. When a
failure happens, the replay string is emitted so you can reproduce the case
directly.

## Minimization and Reduction

After a failure, chaos_theory minimizes the trace:

- It removes or shortens entire scopes when possible.
- It minimizes numeric choices using binary-search-like strategies.
- It keeps the property failing while making the trace smaller.

Labels and `Effect` allow this to be done efficiently.

## Mutation and Crossover

For fuzzing, the trace is treated as structured input. The mutator performs
structure-aware changes rather than blind byte-level flips. Crossover combines
two traces while preserving their internal structure.

This is the core of structure-aware fuzzing in chaos_theory.

## Swarm Testing

Swarm testing is about exploring different subspaces by selectively enabling
only some choices per run. It avoids exploring the full combinatorial space all
at once and often finds bugs faster.

In chaos_theory, this is built directly into choice selection. Roughly half the
time, a choice is made from a smaller bound (a random subset size), then
deterministically permuted back into the full range. Across runs, this yields
different subsets of choices and a broad exploration without a special "swarm
mode".

## Budget and Temperature

Two runtime controls influence generation:

- Budget limits how much randomness can be consumed per test case.
- Temperature biases the randomness distribution.

These exist to keep generation bounded and to allow deliberate exploration strategies.

## Logging and Replay Output

Logging is scoped and tied to `Source::should_log`. By default, logs only appear
for the failing case. Log depth determines how deep the scope stack must be for
logs to show up, which keeps output focused and fast.
