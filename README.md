# chaos_theory

chaos_theory is a modern Rust property-based testing and structure-aware fuzzing library.

## Features

- Property-based testing *and* structure-aware fuzzing support
- Advanced functionality, including:
  - Data generation biased to explore edge cases
  - Built-in universal swarm testing
  - Seeded generation
- Simple, imperative, macro-free API
- Zero unsafe code
- Zero required dependencies

## Documentation

API documentation: [docs.rs/chaos_theory](https://docs.rs/chaos_theory)

## Status

chaos_theory is pretty good, and is widely relied upon in our internal codebase.
However, some important functionality (derive macro, proper recursion handling,
NaN generation) is missing, and there is no documentation besides minimalistic docstrings.

chaos_theory has not been officially released yet, and is certainly lacking
the required polish. Use at your own risk and don't expect support.

## License

chaos_theory is licensed under the [Mozilla Public License Version 2.0](./LICENSE).
