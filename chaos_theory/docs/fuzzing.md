# Fuzzing

Fuzzing should just be another way to drive a property you already test, not a
separate fuzz-only property. `chaos_theory` is responsible for structured
generation and mutation, while [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz)
and libFuzzer provide the coverage-guided loop and corpus management.

## Set Up `cargo-fuzz`

Install `cargo-fuzz` and initialize its directory:

```console
rustup toolchain install nightly --profile minimal
cargo install cargo-fuzz
cargo fuzz init
```

In the generated `fuzz/Cargo.toml`, replace the generated dependency sections
with the `chaos_theory_libfuzzer` fork under the usual `libfuzzer-sys`
dependency name:

```toml
[dependencies]
libfuzzer-sys = { package = "chaos_theory_libfuzzer", version = "0.4.13" }
chaos_theory = "0.3"
my_crate = { path = ".." }
```

Extract the closure normally passed to `check` into a public function, then
replace the generated fuzz target with:

```rust,ignore
#![no_main]

// In `my_crate`:
// pub fn my_prop(src: &mut chaos_theory::Source) { ... }

chaos_theory::fuzz_target_libfuzzer!(my_crate::my_prop);
```

## Run

From the package root, start fuzzing:

```console
cargo +nightly fuzz run fuzz_target_1
```

For ordinary property failures, `chaos_theory` prints a
`CHAOS_THEORY_REPLAY=...` value. Apply it to the regular test that calls
[`check`][check] with your property to reproduce and minimize the failure
outside the fuzzing process. Native crashes remain available as normal
cargo-fuzz artifacts.

## Why The Fork?

In short – because `chaos_theory` is an immediate-mode structural fuzzing API,
which is quite unusual (unique?) in the fuzzing world. See the
[`chaos_theory_libfuzzer`](https://crates.io/crates/chaos_theory_libfuzzer)
README for a more detailed explanation.

[check]: crate::check
