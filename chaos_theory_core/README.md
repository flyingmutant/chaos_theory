# chaos_theory core

## a plan?

- hard mode first: context-dependent generation
- lean into forward mode
- "what is happening" for obs
  - trace-first
    - create/enter/exit span with a label
    - wha
  - cmplog-friendly
  - allow to query where we are in the trace (including globally, across many runs)
    - the coordinates should allow to encode stateful generation
    - encode entity creation, reset etc., choice from a set with a stable key
  - use this knowledge to generate pseudo-random values
    - auto-regressively work with previously generated trace parts
- composable specification of behavior
  - data (shape) + functions (stateless) + classes (stateful)
- generate libraries for all languages (except rust)

## new ideas

- skip "value"/generator things almost completely; the core is pseudo-random process control (that may not generate anything, really)
- lean maximally into LLM-friendly "this is just a smart PRNG you use to do random choices" design
  - can you infer scopes from the source code when doing minimization somehow?
  - make the thing as forward-mode-generation-only as possible, simplest possible mental model wins
- unite Source/SourceRaw, make examples entirely optional
- generator is a bad/wrong abstraction -- people *don't* want to write generators
  - but people *do* want convenient data generation helpers

## goals

- language-agnostic core with no dependencies
- allow building, as one coherent frontend
  - programmable tracing fuzzer
    - property-testing
      - deep structural semantic mutations (repeat/select/...)
    - fuzzing
      - with no compiler instrumentation necessary
    - observability (= testing in production)
      - faithfully represent high-level control (including recursion) & data flow
      - faithfully represent concurrency/parallelism/async
      - can be cleanly and totally disabled, statically or dynamically
  - structural data generator
    - faithfully represent nested structs/enums
    - biased generation
      - swarm testing
    - stateful (context-dependent) generation
    - example-based generation
  - library for building composable specifications of behavior
    - think core.spec
- start from the bits up
  - represent rich choice, represent repetition, represent rejection sampling
    - choice is dual of repetition (we choose from a repetition)
  - represent coordinates (& recursion); attach things to scopes
  - represent what we would want to explore and what have we explored (coverage)
  - represent `hash-of-this-trace-of-state-space` thing, to explore efficiently
  - represent custom feedback (hit an interesting branch/value)
  - represent normal/error control flow
  - represent dual compose (generate) / decompose (example-to-tape)
  - represent "new entity" creation & "reset", & penalize them
  - represent "an interesting trace" (a ~bug, together with approximate cause)
  - we have to be able to mirror all control flow with high precision
- implementation
  - be prepared for sloppy traces
  - allow integrating solvers/oracles
  - lifetime-free API

## ideas

- we have a current path, induction: add 1 bit
  - want to request state space explosion (doubling)
    - repeatedly, in case of N bits
  - want to label each state space path (0 vs 1)
  - axis: continue (to the right)
  - axis: go deeper (down)
  - path-to-bit: path + segment-horizontal (exactly one)
  - path-inside-bit: segment-horizontal (up to one)
- multi-stage call/return? capture both caller & callee perspectives
- what we are doing is reified control flow (+ exec history) enriched with intent
