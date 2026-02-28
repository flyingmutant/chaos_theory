# chaos_theory

chaos_theory is a modern Rust property-based testing and structure-aware fuzzing library.

You drive tests using `Source` to get structured pseudo-random values and control flow,
chaos_theory ensures that this exploration is efficient and any failures found
are automatically minimized.

[![crates.io](https://img.shields.io/crates/v/chaos_theory.svg)](https://crates.io/crates/chaos_theory)
[![docs.rs](https://img.shields.io/docsrs/chaos_theory)](https://docs.rs/chaos_theory)

## Quickstart

```rust
use chaos_theory::{check, make::string_matching};

#[test]
fn slug_and_id_roundtrip() {
    check(|src| {
        let slug = src.any_of("slug", string_matching("[a-z0-9]+(-[a-z0-9]+)*", true));
        let id: u32 = src.any("id");

        let s = format!("{slug}-{id}");
        let (slug_parsed, id_parsed) = s
            .rsplit_once('-')
            .map(|(s, i)| (s, i.parse::<u32>().unwrap()))
            .unwrap();

        assert_eq!(slug_parsed, slug);
        assert_eq!(id_parsed, id);
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
- Macro-free, imperative API (with optional derive macro for `Arbitrary`)
- Zero unsafe code and zero required dependencies

## Documentation

https://docs.rs/chaos_theory

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
