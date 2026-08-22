# Changelog

## Unreleased

- Rename `*_with_size` to `*_n` and `*_in_range` to `*_in`.
- Rename `SourceRaw` to `SourceEx`.
- Report likely hanging check iterations and their seeds, as well as hanging replay/reduction runs.
- Add a `filter` generator modifier to `#[derive(Arbitrary)]` for structs/enums/fields.
- Rename `Generator::{filter_assume, filter}` to `Generator::{filter, try_filter}`.
- Rename `make::{from_fn_assume, from_next_assume}` to `make::{from_fn_some, from_next_some}`.
- Make `make::index` produce `usize` and panic for an empty range; add `make::try_index`.

## 0.4.0 (2026-08-19)

- Switch libFuzzer backend to the `chaos_theory_libfuzzer` fork.
- Rename `fuzz_target_libfuzzer_sys!` to `fuzz_target_libfuzzer!`.
- Drop `fuzz_write_seed` (it is no longer required).
- Add fuzzing mini-guide.

## 0.3.8 (2026-08-15)

- Add feature-gated `either` support with `make::either::{either, left, right}` and `Arbitrary` impl.
- Add `Arbitrary` impls for IP and socket address types.
- Add regular expression generators for `CString` and owned `CStr` types.
- Fix the `no_std + regex` build.
- Update to Unicode 17.

## 0.3.7 (2026-08-14)

- Make float `Arbitrary` be able to generate all possible bit patterns (including NaNs and -0).
- Add feature-gated `jiff` support (`SignedDuration`, `Timestamp`, `TimeZone`, and `Zoned`).

## 0.3.6 (2026-08-11)

- Add `Arbitrary` impls for 128-bit integers.
- Add `Arbitrary` impls for `PathBuf` and owned `Path` types.
- Add target-specific `Arbitrary` impl for `OSString`.
- Add `CString` generators.
- Add trivial `Arbitrary` impl for `MaybeUninit`.
- Add `Arbitrary` impls for `Range` and `Bound` types.
- Add `should_log()`, remove `Source::should_log`, make `vdbg!`, `vprintln!`, and `veprintln!` usable everywhere.

## 0.3.5 (2026-03-25)

- Fix broken `chaos_theory_derive` dependency version.

## 0.3.4 (2026-03-25)

- Add feature-gated `bstr` support with `make::bstr::{bstring, bstring_with_size}` and `Arbitrary` impl.
- Add feature-gated `bytes` support with `make::bytes::{bytes, bytes_mut}` and `Arbitrary` impls.
- Add feature-gated `ordermap` support with `make::ordermap::{order_map, order_set}` and `Arbitrary` impls.
- Add feature-gated `uuid` support with `make::uuid::uuid_v4()` and `Arbitrary for uuid::Uuid` (v4 only).
- Rename `from_fn` to `from_next` and add simpler `from_fn` that ignores examples

## 0.3.3 (2026-03-23)

- Make `std` and `no_std` compose: enabling both now builds the `std` variant instead of hard-failing,
  which lets `no_std` libraries keep `chaos_theory` arbitrary support in normal dependencies while
  tests or downstream crates enable `std`.

## 0.3.2 (2026-03-21)

- Add support for `#[chaos_theory(generator = EXPR)]` attribute for fields in derive macro
- Better `no_std` mode

## 0.3.1 (2026-03-18)

- Add support for property determinism checking (advisory by default, enforce with `Config::with_check_determinism`)
- Add feature-gated `std` support (`std` is default-on) and a `no_std` feature for `no_std + alloc` builds.
  - Support generation APIs and `#[derive(chaos_theory::Arbitrary)]` in `no_std + alloc`.
  - Keep checking, fuzzing, sync primitives, and `SystemTime` generators `std`-only; in `no_std`, `Config::env` ignores environment variables.
  - Add deterministic `no_std` default seeding with `jump_seed_sequence`, and add CI compile checks for `no_std` and `no_std + derive`.

## 0.3.0 (2026-02-28)

- Add optional `derive` feature with `#[derive(chaos_theory::Arbitrary)]` support.

## 0.2.2 (2026-02-23)

- Make FAQ link to API items.

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
