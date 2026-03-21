# Guide

This is a short guide to chaos_theory basics. It covers how to write properties with
[`Source`][source], with a focus on the two core building blocks after [`any`][source_any]:
[`repeat`][source_repeat] and [`select`][source_select], and includes an overview of generators.

*Note: original version of this document was AI-generated.*

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

[`Source`][source] gives you structured randomness. Your job is to explore the system using it.

## `no_std` Usage

`chaos_theory` enables `std` by default. To use it in `no_std + alloc`, disable defaults:

```toml
[dependencies]
chaos_theory = { version = "0.3", default-features = false, features = ["no_std", "derive"] }
```

In `no_std`, generation APIs are available (`Arbitrary`, `Generator`, `Source`, `SourceRaw`,
`Env::example`, `make::*` core/alloc generators), while `check` and fuzzing APIs remain `std`-only.
`Config::env(true)` does not read environment variables in `no_std`.
Default seeding is deterministic there and can be advanced with `jump_seed_sequence`.

## Working With `Source`

The basic operations:

- [`any`][source_any] / [`any_of`][source_any_of] generate values from [`Arbitrary`][arbitrary]
  or from a generator.
- [`choose`][source_choose] selects an element from a slice.
- [`select`][source_select] chooses a labeled variant and runs a branch.
- [`repeat`][source_repeat] repeats a step, using [`Effect`][effect] to report what happened.
- [`maybe`][source_maybe] and [`find`][source_find] are for optional steps and recoverable failures.

Labels matter. Use short, stable labels (like `"action"` or `"key"`). They are
not required for chaos_theory to work, but they make replay output and
reproduction steps readable. With good labels, the failing case description is
often enough to spot the issue immediately.

## `select`: Variants With Meaning

[`select`][source_select] is how you define structured choices:

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
[`select`][source_select] so replay and minimization can preserve the variant choice
when necessary.

## `repeat`: Exploration Over Time

[`repeat`][source_repeat] is the right way to explore sequences:

```rust
use chaos_theory::Effect;

# fn prop(src: &mut chaos_theory::Source) {
src.repeat("step", |src| {
    // perform one step
    Effect::Success
});
# }
```

[`Effect`][effect] matters:

- [`Success`][effect_success]: useful work was done.
- [`Change`][effect_change]: state may have changed, but no clear progress.
- [`Noop`][effect_noop]: nothing happened to the system (example: action was non-applicable).

Honest [`Effect`][effect] values make exploration and minimization much more efficient.

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

Use [`repeat`][source_repeat] instead. `repeat` is structured and minimizes
well, while manual random loops are opaque and minimize poorly.

Another version of the same issue is:

```rust
# fn prop(src: &mut chaos_theory::Source) {
let do_it: bool = src.any("do_it");
if do_it { /* use src here */ }
# }
```

Prefer [`maybe`][source_maybe] or [`select`][source_select] so the execution shape is tracked
structurally.

## Stateful Testing (State Machines)

chaos_theory does not have a special API for state machines. The normal API is
already the "advanced" mode.

The most common pattern is:

1. Build the system under test and a reference model.
2. [`repeat`][source_repeat] a step that selects and applies an action.
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

Nested [`repeat`][source_repeat] and [`select`][source_select] are normal and encouraged
for complex stateful systems.

## Filtering And Validity

If you need to reject invalid values, prefer recoverable filtering:

- [`Generator::filter`][generator_filter] returns `Option` so you can handle failure
  without panicking.
- [`filter_assume`][generator_filter_assume] and [`assume!`][assume] mark the whole test case as
  invalid when the condition fails.

Too many invalid cases will make [`check`][check] fail early because it cannot generate enough
valid tests.

## Debugging Output

Use [`should_log`][source_should_log], [`vdbg!`][vdbg], and [`vprintln!`][vprintln] so output
appears only for the failing case. It keeps tests fast and logs focused.

## Generators

Most users never write custom generators. You can get far with:

- built-in generators in [`make::*`][make],
- composing with [`select`][source_select], [`repeat`][source_repeat], and [`any`][source_any],
- occasional use of [`from_fn`][make_from_fn] if needed.

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

- Core: [`just`][make_just], [`one_of`][make_one_of], [`mix_of`][make_mix_of],
  [`option`][make_option], [`result`][make_result]
- Numbers: [`int_in_range`][make_int_in_range], [`float_in_range`][make_float_in_range]
- Strings and chars: [`string`][make_string], [`string_with_size`][make_string_with_size],
  [`char_ascii`][make_char_ascii], [`byte_ascii`][make_byte_ascii]
- Collections: [`vec`][make_vec], [`vec_with_size`][make_vec_with_size],
  [`btree_map`][make_btree_map], [`hash_map`][make_hash_map],
  [`btree_set`][make_btree_set], [`hash_set`][make_hash_set]
- Time: [`duration_in_range`][make_duration_in_range],
  [`system_time_in_range`][make_system_time_in_range]
- Sync and cells: [`mutex`][make_mutex], [`rw_lock`][make_rw_lock],
  [`once_lock`][make_once_lock], [`cell`][make_cell], [`ref_cell`][make_ref_cell]
- Regex (feature-gated): [`string_matching`][make_string_matching],
  [`bytes_matching`][make_bytes_matching]
- Extra crates (feature-gated): [`hashbrown`][make_hashbrown], [`indexmap`][make_indexmap],
  [`ordered_float`][make_ordered_float], [`tinyvec`][make_tinyvec], [`ecow`][make_ecow]

If a generator exists, prefer using it instead of re-implementing the logic.

### Default Distribution

chaos_theory is designed to give you a useful distribution by default. Numeric
choices are heavily biased toward small values, built-in generators include
curated seed values, boundary values are prioritized, and swarm testing is always on.
The goal is smart data generation that explores the state space quickly
without you having to tune distributions by hand.

### Composing Generators

Useful combinators:

- [`map`][generator_map] and [`map_reversible`][generator_map_reversible] for transforms
- [`or`][generator_or] and [`mix_of`][make_mix_of] for alternatives
- [`collect`][generator_collect] and [`collect_n`][generator_collect_n] for collections
- [`and_then`][generator_and_then] for flat-map style composition

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

Built-in generators already have seeds pre-configured internally, so use [`seeded`][generator_seeded]
only to provide seeds that are specific to your domain.

### Deriving `Arbitrary`

For most domain types, prefer derive instead of hand-writing generators.

Enable derive support:

```toml
[dev-dependencies]
chaos_theory = { version = "0.3", features = ["derive"] }
```

Then derive:

```rust
# #[cfg(feature = "derive")]
#[derive(Debug, chaos_theory::Arbitrary)]
struct Point {
    x: i32,
    y: i32,
}
```

Derive-generated implementations follow the same rules as hand-written ones:
variant choices are structural (`select`) and `example` is threaded through fields.

For field-level customization, use `#[chaos_theory(generator = ...)]` on a struct field
or on the payload field of an enum variant. The expression must evaluate to a
[`Generator`][generator] for that field type:

```rust
# #[cfg(feature = "derive")]
use chaos_theory::{Generator, make};

fn odd_u8() -> impl Generator<Item = u8> {
    make::from_fn(|src, example| src.any::<u8>("base", example) | 1)
}

#[derive(Debug, chaos_theory::Arbitrary)]
enum Op {
    Reset,
    Set(#[chaos_theory(generator = odd_u8())] u8),
    Shift {
        #[chaos_theory(generator = make::just(0))]
        by: i16,
    },
}
```

### Writing Custom Generators

There are two main approaches:

- Use [`make::from_fn`][make_from_fn] for small generators
- Implement [`Generator`][generator] directly for full control

With derive support for [`Arbitrary`][arbitrary], most of this is unnecessary for plain data
models, but it is still useful for domain-specific logic.

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

Use [`select`][sourceraw_select] to choose a variant with a stable label:

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

Use [`repeat`][sourceraw_repeat] to build the collection:

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

Avoid calling [`Generator::next`][generator_next] directly. Use [`Source::any`][source_any] or
[`Source::any_of`][source_any_of] instead.

[source]: crate::Source
[source_any]: crate::Source::any
[source_any_of]: crate::Source::any_of
[source_choose]: crate::Source::choose
[source_select]: crate::Source::select
[source_repeat]: crate::Source::repeat
[source_maybe]: crate::Source::maybe
[source_find]: crate::Source::find
[source_should_log]: crate::Source::should_log
[sourceraw_select]: crate::SourceRaw::select
[sourceraw_repeat]: crate::SourceRaw::repeat

[effect]: crate::Effect
[effect_success]: crate::Effect::Success
[effect_change]: crate::Effect::Change
[effect_noop]: crate::Effect::Noop

[arbitrary]: crate::Arbitrary
[generator]: crate::Generator
[generator_filter]: crate::Generator::filter
[generator_filter_assume]: crate::Generator::filter_assume
[generator_map]: crate::Generator::map
[generator_map_reversible]: crate::Generator::map_reversible
[generator_or]: crate::Generator::or
[generator_collect]: crate::Generator::collect
[generator_collect_n]: crate::Generator::collect_n
[generator_and_then]: crate::Generator::and_then
[generator_seeded]: crate::Generator::seeded
[generator_next]: crate::Generator::next

[check]: crate::check
[assume]: crate::assume
[vdbg]: crate::vdbg
[vprintln]: crate::vprintln

[make]: crate::make
[make_from_fn]: crate::make::from_fn
[make_mix_of]: crate::make::mix_of
[make_just]: crate::make::just
[make_one_of]: crate::make::one_of
[make_option]: crate::make::option
[make_result]: crate::make::result
[make_int_in_range]: crate::make::int_in_range
[make_float_in_range]: crate::make::float_in_range
[make_string]: crate::make::string
[make_string_with_size]: crate::make::string_with_size
[make_char_ascii]: crate::make::char_ascii
[make_byte_ascii]: crate::make::byte_ascii
[make_vec]: crate::make::vec
[make_vec_with_size]: crate::make::vec_with_size
[make_btree_map]: crate::make::btree_map
[make_hash_map]: crate::make::hash_map
[make_btree_set]: crate::make::btree_set
[make_hash_set]: crate::make::hash_set
[make_duration_in_range]: crate::make::duration_in_range
[make_system_time_in_range]: crate::make::system_time_in_range
[make_mutex]: crate::make::mutex
[make_rw_lock]: crate::make::rw_lock
[make_once_lock]: crate::make::once_lock
[make_cell]: crate::make::cell
[make_ref_cell]: crate::make::ref_cell
[make_string_matching]: crate::make::string_matching
[make_bytes_matching]: crate::make::bytes_matching
[make_hashbrown]: crate::make::hashbrown
[make_indexmap]: crate::make::indexmap
[make_ordered_float]: crate::make::ordered_float
[make_tinyvec]: crate::make::tinyvec
[make_ecow]: crate::make::ecow
