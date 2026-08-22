# TODO

## docs

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

## features

- consider `reflecting` as a feature (not just a seed)
- consider runs with forced boundary conditions
  - can swarming with choice = free-gen vs boundary-gen give this for free?

## internals

- fuzz smoke in CI
- consider using `core::range::Range` (requires 1.96)
- consider using `assert_matches!` (requires 1.96)
