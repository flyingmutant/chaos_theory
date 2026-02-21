# chaos_theory

chaos_theory is a modern Rust property-based testing and structure-aware fuzzing library.

You drive tests using `Source` to get pseudo-random values and control flow;
chaos_theory records choices, their structure, and metadata to efficiently explore
the state space and automatically minimize any failures found.

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

When a failure happens, chaos_theory prints a `CHAOS_THEORY_REPLAY=...` string
you can use to reproduce the (typically already minimized) case.

## Highlights

- Property testing and structure-aware fuzzing in one library
- Efficient state space exploration:
  - bias towards small values and edge cases
  - structural mutations and crossover
  - example-guided generation
  - built-in swarm testing
- Macro-free, imperative API
- Zero unsafe code and zero required dependencies

## Documentation

- API docs: https://docs.rs/chaos_theory
- Guide: [docs/guide.md](./docs/guide.md)
- Generators: [docs/generators.md](./docs/generators.md)
- Configuration: [docs/config.md](./docs/config.md)
- Fuzzing: [docs/fuzzing.md](./docs/fuzzing.md)
- Internals: [docs/internals.md](./docs/internals.md)
- FAQ: [docs/FAQ.md](./docs/FAQ.md)

## Status

chaos_theory is used internally and already useful, but it's not officially released yet.

Notable gaps:
- Derive macro for `Arbitrary`
- Proper recursive generators

## License

MPL-2.0, see [LICENSE](./LICENSE).
