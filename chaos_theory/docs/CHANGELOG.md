# Changelog

## Unreleased

- Nothing yet.

## 0.2.1 (2026-02-23)

- Move all documentation over to the crate docs.

## 0.2.0 (2026-02-23)

- Documentation polish.

## 0.1.10 (2026-02-19)

- Add `make::token()`.

## 0.1.9 (2026-02-19)

- Fix repeat mask size for noop elements.

## 0.1.8 (2026-02-19)

- Fix meta-related "test result changed after removing data marked as noop" false positives.
- Track discardable noop repeat elements.
- Avoid repeat element effect upgrades in crossover.
- Major documentation expansion (guide, generators, config, fuzzing, internals, FAQ, README).

## 0.1.7 (2026-01-25)

- Add `ecow` support.
- Add `Σ` and `__class__` to built-in seeds.
- Add docs.rs feature annotations to `Arbitrary` impls.

## 0.1.6 (2025-10-21)

- Add `tinyvec` and `ordered_float` support.
- Avoid a possible overflow.
- Honor zero trials when asked not to reduce.

## 0.1.4 (2025-10-12)

- Add Saturating and Wrapping generators.
- Add `hashbrown` and `indexmap` support.
- Improve recursive data generation.
- Document items that require feature flags.

## 0.1.0 (2025-07-23)

- Initial public release of `chaos_theory`.
