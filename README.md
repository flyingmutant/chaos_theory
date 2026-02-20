# chaos_theory

chaos_theory is a modern Rust property-based testing and structure-aware fuzzing library.

It's built around a simple idea: a `Source` produces structured randomness, and chaos_theory records the exact choices so failures can be reproduced, minimized, and mutated.

## Quickstart

```rust
use chaos_theory::check;

#[test]
fn sort_is_idempotent() {
    check(|src| {
        let mut v: Vec<i32> = src.any("v");
        let mut w = v.clone();
        v.sort();
        w.sort();
        assert_eq!(v, w);
    });
}
```

When a failure happens, chaos_theory prints a `CHAOS_THEORY_REPLAY=...` string you can use to reproduce the (typically already minimized) case.

## Highlights

- Property testing and structure-aware fuzzing in one library
- Smart biased generation (small values, boundary values and built‑in seeds).
- Example-guided generation and seeded inputs
- Universal swarm testing for exploration
- Macro-free, imperative API
- Zero unsafe code and zero required dependencies

## Fuzzing Quickstart

```rust
use chaos_theory::fuzz_target_libfuzzer_sys;

fuzz_target_libfuzzer_sys!(|src| {
    let v: Vec<u8> = src.any("v");
    // your invariants here
});
```

Before fuzzing, generate seeds with `chaos_theory::fuzz_write_seed`.

## Documentation

- API docs: https://docs.rs/chaos_theory
- Guide: `docs/guide.md`
- Generators: `docs/generators.md`
- Config and repro: `docs/config.md`
- Fuzzing: `docs/fuzzing.md`
- Internals: `docs/internals.md`
- FAQ: `docs/FAQ.md`

## Status

chaos_theory is used internally and already useful, but it's not officially released yet.

Notable gaps:
- Derive macro for `Arbitrary`
- Proper recursive generators

## License

MPL-2.0, see `LICENSE`.
