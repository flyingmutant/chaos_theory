# Generators

This document is about how to use and (when needed) write generators. Most
people get far with built-ins and composition, but custom generators are
absolutely fine when you need them.

## In Practice

You will spend most of your time doing this:

```rust
let v: Vec<String> = src.any("v");
```

Custom generators are useful for domain types or complex invariants, but they
are not required for everyday property tests.

## Built-Ins (`make::*`)

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

## Default Distribution

chaos_theory is designed to give you a useful distribution by default. Numeric
choices are heavily biased toward small values, built‑in generators include
curated seed values, boundary values are prioritized, and swarm testing is always on.
The goal is smart data generation that explores the state space quickly
without you having to tune distributions by hand.

## Composing Generators

Useful combinators:

- `map` and `map_reversible` for transforms
- `or` and `mix_of` for alternatives
- `collect` and `collect_n` for collections
- `and_then` for flat-map style composition

## Seeded Generation

Use seeds when you have real examples that should guide exploration:

```rust
use chaos_theory::make;

let cities = ["Tokyo".to_owned(), "Moscow".to_owned(), "Shanghai".to_owned()];
let city = make::string_matching("[A-Za-z '-]+", true).seeded(&cities, true);
```

Built-in generators already have seeds pre-configured internally,
so use `seeded` only to provide seeds that are specific to your domain.

## Filtering And Validity

Prefer recoverable filtering:

- `Generator::filter` returns `Option` you can handle
- `filter_assume` and `assume!` can mark the whole test case as invalid on failure

Too many invalid cases will make `check` fail early.

## Writing Custom Generators

There are two main approaches:

- Use `make::from_fn` for small generators
- Implement `Generator` directly for full control

Most of this becomes unnecessary once a derive macro for `Arbitrary` exists, but
it is still useful for domain-specific logic.

### Struct-Like Types

The pattern is “generate fields, then build the struct”. Always pass field examples when present.

```rust
use chaos_theory::{Generator, SourceRaw};

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

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

### Enum-Like Types

Use `select` to choose a variant with a stable label:

```rust
use core::num::NonZero;
use chaos_theory::{Generator, SourceRaw};

#[derive(Debug)]
enum Op {
    Add(i32),
    Reset,
}

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

### Collection-Like Types

Use `repeat` to build the collection:

```rust
use chaos_theory::{Arbitrary, Effect, Generator, SourceRaw};

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
