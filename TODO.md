# TODO

## OSS public release

- rename tape to trace?
- warn on `as` conversions
- usage examples
- grep TODO
- consider renaming `_with_size` to `_n`
- `no_std` so that people can define generators without `std`
- expose generator types?
- `src.any("").filter()` API want? `choose_where` looks simple, `any_of + filter` does not
  - `any` == `choose` from whatever!
- consider `make::string` + `make::string_of`
  - dot works better + type inference is simpler + python is simpler
  - what to do with `int*` and `float*`?
- generator-centric API, with `src` becoming `rand` argument?
- want to create special-case so that `choose` on mutable collections reduces nicely
- `choose`/`select` stable key
  - special `Hash`-like trait or just hash?
- `minimize` API with explicit goal that returns the minimum value

## Intelligent generation

- penalize "new entity" creation actions (they increase the state we are testing)
  - maybe: track the current "size stack" (repeat nesting and sizes), and penalize new entity creation in repeat as well?
    this should generalize to a simple algorithm that generates compact nested structures,
    and maybe even handles recursion automatically
- penalize "reset" actions (they nullify testing efficiency often)
- custom feedback for fuzzer that estimates operation diversity, flow diversity, parameter diversity
  - more detailed `select`/`repeat` feedback: "new entry created" / "subsystem reset" / "error"
- this all should be somehow integrated into `Effect`
  - or maybe some helpers like `src.can_create()`, `src.can_reset()`?

## General

- cover
- reldata
- nice-to-have
  - flaky test detection
  - collect N bugs during one run
    - MoreBugs approach: avoid hitting the same bug based on the generalized bug signature
  - make sure debug assertions are not triggered when fuzzing (rustc flag `-C debug-assertions`)
    - probably need to make special assert macro for that
  - evaluate if we can omit `src` parameter from `vdbg` and friends (thread-local?) for comfortable debugging (also add indent)
    - thread-local will be nice for `fail!` later (unless sans-io is used, then it provides no benefit)
    - can we generalize this so we can insert `assert_sometimes` inside as well?
  - show when the value does not matter for the failure (as a `// value does not matter` comment during logging)
- improved fundamentals
  - have generators optionally accept `&mut ctx` (and `&mut example_ctx`), to support stateful generation
    and generation of conditionally valid things (think tape event, or SQL sub-expression)
  - auto-register events for each branch of `select`/`maybe` and zero/one/multiple `repeat` iterations (if possible);
    keep generating data until we've hit all the registered cases
    - need to think, what about conditionally-valid branches/generators, is "full coverage" impossible in general?
  - hypothesis shrinks bundles towards right (to make dependencies shorter); should we do the same for `choose`?
  - sometimes (rarely), generate values of more than `MAX_SIZE`
    - maybe warn about explicit upper size bound being specified that is larger than `MAX_SIZE` (because it will never be generated)
  - mark scopes of primitive generators as such to improve mutation efficiency
    - also, by associating generators with scopes we can run generators during mutation and re-use interesting values & other heuristics
  - `choose` operates on indices which are unstable (e.g. during minimization);
    it would be nice to bound `choose` directly to the stable *value* that is chosen
    (something like stable primary key in tables)
  - for void scopes, there is a problem that the sequence we'll generate for a seed is always the same.
    in particular, during reduction, we are relatively likely to generate tapes like PREFIX-WHATEVER-PREFIX.
    maybe we should start void #X with the seed that depends on base seed and X?
    this will also make different void fragments not affect each other based on their size (= amount of RNG used)
  - for `seeded`, don't only reuse the top-level tapes: have a possibility of reuse at each level (partial reuse)
    - maybe we can aggressively void-out parts of the tape; 0 = full reuse & full void = new value
  - single `Noop` `Effect`, which we discard intellectually
  - recursive generators support
    - either hacks based on `new_cyclic` that increases the recursion depth during `next`,
    - or an approach like hypothesis `recursive` which creates logarithmic number of pre-composed generators + rejection-filters the ones that generate too many leaves
    - or a combined approach: recursion-depth hacks + rejection-filtering
    - need to think about IDs: we want our mutator to be able to replace whole with subparts
      - probably like `repeat` gives identical IDs to elements, recursive combinator should give identical IDs to parts as well
    - maybe use depth as well as scope IDs for swarm testing etc. to handle recursion
  - flatten `or` and `mix_of` somehow (draw from the flattened list of generators)
  - find a way to unify int/byte generation:
    - use flatter "small values" for all integers, without special-casing `u8`
    - also, fall `choose_value` back to using `choose_index` for small values
      - make `byte_ascii()` be literally `int_in_range(..=0x7f)`
  - more semantics generation: biased `select`
    - problem: select bias might be a dynamic thing: consider any FSM-like API
    - maybe just have a way to nop-exit repeat step when no budget to run rare ops
    - maybe do this via some kind of smart precondition?
    - what kind of information to attach: something error/reset-like?
      - "hint::error", "hint::reset"
      - how many groups, what about `f64` and the like?
    - when to attach, before-hand or imperatively during execution?
    - make sure nesting works well
- additional docs
  - general usage
    - note about logging in tests
    - note about determinism and flaky-ness
  - README
  - FAQ
  - internals
    - where generator state is stored and why (generators don't live between check iterations)
      - state is associated with scopes (`repeat` etc, they are kind of not really generators)
  - examples
- more generators
  - u128
  - CString/Cstr
  - OSString/OSstr
  - PathBuf/Path
  - IpAddr/SocketAddr
  - range types, Bound
  - MaybeUninit
  - permutation/shuffle/subsequence
  - recursive
    - `nothing` generator that calls `reject` (e.g. to specify desired tree structure?)
- derive crate
- open source release

## Ideas

- for compatibility with binary fuzzer cmplog mutations:
  - store in the tape binary representation of the mutated value
  - if we mutated it, put it on the corresponding scope start and use as an example
  - will need a `CanonicalBinary` trait or something like that which we'll implement for all simple types
- in `select`, can use syntax inside `"action_name"` to mark things as rare
  - roll, then see the name. if name is rare and we have no budget, re-roll
- provide `&impl Hash` thing to explore state space
- something like `if s.fail("label")` (see [fail crate](https://docs.rs/fail/latest/fail/))
  - important, real-world case
  - how to thread `s` into regular user code? don't want code to depend on `chaos_theory` probably
- provide `likely!` and `unlikely!` macros (functions?) to help with generation
- reuse raw bytes, somehow
  - string <-> int <-> []byte; do we get this from the tape?
 . - create higher-level data structure, then transfer to lower level (or even bytes), and corrupt/mutate the lower-level
- instead of cmplog, provide an explicit "this value may be of interest" construct with much higher signal-to-noise
  - likewise with explicit "sometimes assert" and IJON-like constructs instead of relying on backedge hit counters
    - state space is just too large for most interesting programs to rely on such low-power constructs?
- explore/exploit phases/tradeoffs, based on understanding the "warmth"
  - pso, entropic schedule to guide search
- edge coverage (never-zero), cmplog + result hooking for instrumentation
  - but allow to specify state transitions by hand, and record the order
- increase the temperature when we can't make progress for some time (mutagen paper)
  - magnitude = temperature (is determined by temperature?)
  - new generation = random neighbor of zero
  - neighbor search = random neighbor of prev. value, with specified temperature?
  - we raise the temperature during the generation, first exploring low-entropy states
  - hmm, we need to generate round (in binary) numbers to get bitflips this way
  - we can increase the temperature of the appended random gradually (e.g. we try to generate filter-non-zero from all-zero input)
- LIFO priority queue for inputs
  - lookup A* like algorithms to control this priority queue (we are doing a search, really)
- what we do is explore paths of state changes
  - in simple case this is just a single path
  - in complex cases these are several concurrent paths
  - this generalizes state-machine / stateful testing AND just tracing what has happened during the run
- provide fork-like state exploration (completely in user space)

## Fuzzer design ideas

- log/trace of scopes
- incremental feedback
- expose std api for sensors
- de-prioritize error paths, likely/unlikely
- value-based reuse
- biased select?
- system-of-N-things
- entry selection:
  - start with random, then go to afl-favored
- energy for entry:
  - start with fixed, then go to afl-explore or entropic
- prefix/mutate/suffix approach to mutation
  - first choose entry based on program state, then inside choose based on coverage
- node/target communication:
  - stdin/stdout for commands
  - shmem for data in/out
