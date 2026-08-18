# TODO

## docs

- usage examples
  - update example crates
- skill or one-pager for coding agents
  - iteration control, seeds, hangs

## API

- consider renaming `_with_size` to `_n`
- consider `make::string` + `make::string_of`
  - dot works better + type inference is simpler + python is simpler
  - what to do with `int*` and `float*`?
- expose generator types?

## generators

- permutation/shuffle/subsequence/random chunking
- recursive

## features

- consider saving failures, at least temporarily
- print out potentially-hanging seed (how?)
- consider `reflecting` as a feature (not just a seed)
- consider runs with forced boundary conditions
  - can swarming with choice = free-gen vs boundary-gen give this for free?

## internals

- fuzz smoke in CI
- consider using `core::range::Range` (requires 1.96)
- consider using `assert_matches!` (requires 1.96)
