# Generators

This document is about how to use and (when needed) write generators. Most
people get far with built-ins and composition, but custom generators are
absolutely fine when you need them.

## In Practice

You will spend most of your time doing this:

```rust
let v: Vec<i32> = src.any("v");
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
curated seed values, boundary values are prioritized and swarm testing is always on.
The goal is smart data generation that explores the state space quickly
without you having to tune distributions by hand.

## Composing Generators

Useful combinators:

- `map` and `map_reversible` for transforms
- `or` and `mix_of` for alternatives
- `collect` and `collect_n` for collections
- `and_then` for flat-map style composition

`map_reversible` is the high-quality option: it helps chaos_theory reconstruct
examples and minimize better.

## Seeds And Examples

Use seeds when you have real examples that should guide exploration:

```rust
let seeds = [0u32, 1, 2, 3, 255];
let g = make::int_in_range(0..=255).seeded(&seeds, true);
```

The `example` reference is the reverse direction of generation: it is used to
reconstruct the choice trace that would produce a value. If you ignore it, tests
still work, but replay and minimization get worse.

## Filtering And Validity

Prefer recoverable filtering:

- `Generator::filter` returns `Option`
- `filter_assume` and `assume!` are last resorts

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
#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

struct PointGen;

impl Generator for PointGen {
    type Item = Point;

    fn next(&self, src: &mut SourceRaw, example: Option<&Point>) -> Point {
        let x = i32::arbitrary().next(src, example.map(|e| &e.x));
        let y = i32::arbitrary().next(src, example.map(|e| &e.y));
        Point { x, y }
    }
}
```

### Enum-Like Types

Use `select` to choose a variant with a stable label:

```rust
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
        src.select(
            "<op>",
            example_ix,
            variants.len().try_into().expect("variants"),
            |ix| variants[ix],
            |src, variant, _| match variant {
                "add" => {
                    let ex = match example { Some(Op::Add(v)) => Some(v), _ => None };
                    Op::Add(i32::arbitrary().next(src, ex))
                }
                "reset" => Op::Reset,
                _ => unreachable!(),
            },
        )
    }
}
```

### Collection-Like Types

Use `repeat` to build the collection and report effects honestly:

```rust
struct BytesGen;

impl Generator for BytesGen {
    type Item = Vec<u8>;

    fn next(&self, src: &mut SourceRaw, example: Option<&Vec<u8>>) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0usize;
        src.repeat("<bytes>", example.map(|v| v.iter()), .., |_| ((), 0), |_v, src, ex| {
            let b = u8::arbitrary().next(src, ex);
            out.push(b);
            i += 1;
            Effect::Success
        });
        out
    }
}
```

### Passing `example` Through

The rule is simple: if you generate sub-values, pass the corresponding `example`
sub-values into their generators. This is how chaos_theory reconstructs known
values and minimizes effectively.

Avoid calling `Generator::next` directly outside of generator implementations.
When writing tests, use `Source::any` or `Source::any_of` instead.
