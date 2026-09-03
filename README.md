# chaos_theory

chaos_theory is a modern Rust property-based testing and structure-aware fuzzing library.

You drive tests using `Source` to get structured pseudo-random values and control flow,
chaos_theory ensures that this exploration is efficient and any failures found
are automatically minimized.

[![crates.io](https://img.shields.io/crates/v/chaos_theory.svg)](https://crates.io/crates/chaos_theory)
[![docs.rs](https://img.shields.io/docsrs/chaos_theory)](https://docs.rs/chaos_theory)

## Quickstart

```rust
use chaos_theory::check;

#[test]
fn sort_strings() {
    check(|src| {
        let mut strings: Vec<String> = src.any("strings");
        strings.sort();
        assert!(strings.is_sorted(), "unsorted after sort: {strings:?}");
    });
}
```

When a failure happens, chaos_theory prints a `CHAOS_THEORY_REPLAY=...` string
you can use to reproduce the case.

## Highlights

- Property testing and structure-aware fuzzing in one library
- Efficient state space exploration:
  - bias towards small values and edge cases
  - structural mutations and crossover
  - example-guided generation
  - built-in swarm testing
- Macro-free, immediate-mode API: generate values and control flow as the test runs
- Zero unsafe code, zero required dependencies and `no_std`-compatible

## Documentation

- [API reference](https://docs.rs/chaos_theory)
- [Guide](https://docs.rs/chaos_theory/latest/chaos_theory/_docs/guide/index.html)
- [FAQ](https://docs.rs/chaos_theory/latest/chaos_theory/_docs/faq/index.html)
- [Changelog](https://docs.rs/chaos_theory/latest/chaos_theory/_docs/changelog/index.html)

## Status

Beta – chaos_theory works well and is useful, but does not guarantee API stability.

Notable gaps:
- Proper recursive generators

## Contributing

- Feedback – bug reports, issues, proposals – is welcome
- Pull requests require prior discussion before being open
- All AI usage must be disclosed

## License

MPL-2.0, see [LICENSE](./LICENSE).
