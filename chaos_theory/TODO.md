# TODO

## docs

- README: `regex` feature required for the main example
- advertise structural generation without testing
- better guide text
  - section about how logging works
  - section about common `CHAOS_THEORY_` variables
  - more common anti-patterns
- usage examples
  - update example crates
- skill or one-pager for coding agents
  - iteration control, seeds, hangs

## API

- decide if bool args for `fullmatch` & `mutate_seeds` are OK

## generators

- permutation/shuffle/subsequence/random chunking
- recursive

## internals

- improve the sorting pass (order, sort predicate, swap-around-mid)
  - consider pairs-around-midpoint for main reduction pass as well
- add pair-lowering pass (up to fixed distance, to avoid O(N^2))
- consider runs with forced boundary conditions
  - can swarming with choice = free-gen vs boundary-gen give this for free?
- fuzz smoke in CI
- consider limiting/resetting the caches more during fuzzing for lower memory usage
- consider some lightweight `CHAOS_THEORY_REPLAY` persistence
- consider using `core::range::Range` (requires 1.96)
- consider using `assert_matches!` (requires 1.96)
