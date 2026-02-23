# Guide

This is a short guide to chaos_theory basics. It covers how to write properties with
`Source`, with a focus on the two core building blocks after `any`: `repeat` and `select`,
and includes an overview of generators.

*This is an AI-generated document that was manually reviewed and edited.*

## The Shape Of A Property

Property tests in chaos_theory look like this:

```rust
use chaos_theory::check;

check(|src| {
    // generate structured values
    // drive the system under test
    // assert invariants
});
```

`Source` gives you structured randomness. Your job is to explore the system using it.

## Working With `Source`

The basic operations:

- `any` / `any_of` generate values from `Arbitrary` or from a generator.
- `choose` selects an element from a slice.
- `select` chooses a labeled variant and runs a branch.
- `repeat` repeats a step, using `Effect` to report what happened.
- `maybe` and `find` are for optional steps and recoverable failures.

Labels matter. Use short, stable labels (like `"action"` or `"key"`). They are
not required for chaos_theory to work, but they make replay output and
reproduction steps readable. With good labels, the failing case description is
often enough to spot the issue immediately.

## `select`: Variants With Meaning

`select` is how you define structured choices:

```rust
# fn prop(src: &mut chaos_theory::Source) {
src.select("action", &["insert", "remove", "get"], |src, action, _ix| {
    match action {
        "insert" => { /* ... */ }
        "remove" => { /* ... */ }
        "get" => { /* ... */ }
        _ => unreachable!(),
    }
});
# }
```

You should not encode a variant choice as `any::<u8>()` or a random number. Use
`select` so replay and minimization can preserve the variant choice when necessary.

## `repeat`: Exploration Over Time

`repeat` is the right way to explore sequences:

```rust
use chaos_theory::Effect;

# fn prop(src: &mut chaos_theory::Source) {
src.repeat("step", |src| {
    // perform one step
    Effect::Success
});
# }
```

`Effect` matters:

- `Success`: useful work was done.
- `Change`: state may have changed, but no clear progress.
- `Noop`: nothing happened to the system (example: action was non-applicable).

Honest `Effect` values make exploration and minimization much more efficient.

### Common Anti-Pattern: Manual Random Loops

Don't do this:

```rust
# fn prop(src: &mut chaos_theory::Source) {
let n: usize = src.any("n");
for _ in 0..n {
    /* use src here */
}
# }
```

Use `repeat` instead. `repeat` is structured and minimizes well, while manual
random loops are opaque and minimize poorly.

Another version of the same issue is:

```rust
# fn prop(src: &mut chaos_theory::Source) {
let do_it: bool = src.any("do_it");
if do_it { /* use src here */ }
# }
```

Prefer `maybe` or `select` so the execution shape is tracked structurally.

## Stateful Testing (State Machines)

chaos_theory does not have a special API for state machines. The normal API is
already the "advanced" mode.

The most common pattern is:

1. Build the system under test and a reference model.
2. `repeat` a step that selects and applies an action.
3. Assert invariants or compare against the model.

Example shape:

```rust
# fn prop(src: &mut chaos_theory::Source) {
src.repeat("step", |src| {
    src.select("action", &["insert", "remove", "get"], |src, action, _| {
        // apply action to SUT
        // apply action to model
        // assert invariants
        // return Effect for the chosen action
        # todo!()
    })
});
# }
```

Nested `repeat` and `select` are normal and encouraged for complex stateful systems.

## Filtering And Validity

If you need to reject invalid values, prefer recoverable filtering:

- `Generator::filter` returns `Option` so you can handle failure without panicking.
- `filter_assume` and `assume!` mark the whole test case as invalid when the condition fails.

Too many invalid cases will make `check` fail early because it cannot generate enough valid tests.

## Debugging Output

Use `should_log`, `vdbg!`, and `vprintln!` so output appears only for the
failing case. It keeps tests fast and logs focused.

## Generators

Most users never write custom generators. You can get far with:

- built-in generators in `make::*`,
- composing with `select`, `repeat`, and `any`,
- occasional use of `from_fn` if needed.

Custom generators are useful for domain types or complex invariants, but they
are not required for everyday property tests.

### In Practice

You will spend most of your time doing this:

```rust
# fn prop(src: &mut chaos_theory::Source) {
let v: Vec<String> = src.any("v");
# }
```

### Built-Ins (`make::*`)

Common categories:

- Core: `just`, `one_of`, `mix_of`, `option`, `result`
- Numbers: `int_in_range`, `float_in_range`
- Strings and chars: `string`, `string_with_size`, `char_ascii`, `byte_ascii`
- Collections: `vec`, `vec_with_size`, `btree_map`, `hash_map`, `btree_set`, `hash_set`
- Time: `duration_in_range`, `system_time_in_range`
- Sync and cells: `mutex`, `rw_lock`, `once_lock`, `cell`, `ref_cell`
- Regex (feature-gated): `string_matching`, `bytes_matching`
- Extra crates (feature-gated): `hashbrown`, `indexmap`, `ordered_float`, `tinyvec`, `ecow`

If a generator exists, prefer using it instead of re-implementing the logic.

### Default Distribution

chaos_theory is designed to give you a useful distribution by default. Numeric
choices are heavily biased toward small values, built-in generators include
curated seed values, boundary values are prioritized, and swarm testing is always on.
The goal is smart data generation that explores the state space quickly
without you having to tune distributions by hand.

### Composing Generators

Useful combinators:

- `map` and `map_reversible` for transforms
- `or` and `mix_of` for alternatives
- `collect` and `collect_n` for collections
- `and_then` for flat-map style composition

### Seeded Generation

Use seeds when you have real examples that should guide exploration:

```rust
# use chaos_theory::{make, Source, Generator as _};
# #[cfg(feature = "regex")]
# fn prop(src: &mut Source) {
let cities = ["Tokyo".to_owned(), "Moscow".to_owned(), "Shanghai".to_owned()];
let city = make::string_matching("[A-Za-z '-]+", true).seeded(&cities, true);
# }
```

Built-in generators already have seeds pre-configured internally,
so use `seeded` only to provide seeds that are specific to your domain.

### Writing Custom Generators

There are two main approaches:

- Use `make::from_fn` for small generators
- Implement `Generator` directly for full control

Most of this becomes unnecessary once a derive macro for `Arbitrary` exists, but
it is still useful for domain-specific logic.

#### Struct-Like Types

The pattern is "generate fields, then build the struct". Always pass field examples when present.

```rust
use chaos_theory::{Generator, SourceRaw};

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

#[derive(Debug)]
struct PointGen;

impl Generator for PointGen {
    type Item = Point;

    fn next(&self, src: &mut SourceRaw, example: Option<&Point>) -> Point {
        let x = src.any("x", example.map(|e| &e.x));
        let y = src.any("y", example.map(|e| &e.y));
        Point { x, y }
    }
}
```

#### Enum-Like Types

Use `select` to choose a variant with a stable label:

```rust
use core::num::NonZero;
use chaos_theory::{Generator, SourceRaw};

#[derive(Debug)]
enum Op {
    Add(i32),
    Reset,
}

#[derive(Debug)]
struct OpGen;

impl Generator for OpGen {
    type Item = Op;

    fn next(&self, src: &mut SourceRaw, example: Option<&Op>) -> Op {
        let example_ix = example.map(|e| match e {
            Op::Add(_) => 0,
            Op::Reset => 1,
        });
        let variants = ["add", "reset"];
        let variants_num =
            NonZero::new(variants.len()).expect("internal error: no variants");
        src.select(
            "<op>",
            example_ix,
            variants_num,
            |ix| variants[ix],
            |src, variant, _| match variant {
                "add" => {
                    let v_example = match example { Some(Op::Add(v)) => Some(v), _ => None };
                    let v = src.any("v", v_example);
                    Op::Add(v)
                }
                "reset" => Op::Reset,
                _ => unreachable!(),
            },
        )
    }
}
```

#### Collection-Like Types

Use `repeat` to build the collection:

```rust
use chaos_theory::{Arbitrary, Effect, Generator, SourceRaw};

#[derive(Debug)]
struct BytesGen;

impl Generator for BytesGen {
    type Item = Vec<u8>;

    fn next(&self, src: &mut SourceRaw, example: Option<&Vec<u8>>) -> Vec<u8> {
        let res = src.repeat(
            "<bytes>",
            example.map(IntoIterator::into_iter),
            ..,
            |n| Vec::with_capacity(n),
            |v, src, example| {
                let b = u8::arbitrary().next(src, example);
                v.push(b);
                Effect::Success
            },
        );
        res.expect("bytes repeat must succeed")
    }
}
```

### Passing `example` Through

The `example` reference is the reverse direction of generation:
it is used to reconstruct the choice trace that would produce a value.
This is what makes the seed-based generation work: seeds are converted to a trace,
then this trace, as-is or mutated, is used to produce a seed-based value.

The rule is simple: if you generate sub-values, pass the corresponding `example`
sub-values into their generators. This is how chaos_theory reconstructs known
values and minimizes effectively.

Avoid calling `Generator::next` directly. Use `Source::any` or `Source::any_of` instead.
