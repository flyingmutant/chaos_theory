# Fuzzing

The central idea in chaos_theory is that property testing and fuzzing are the
same thing at different time scales. The properties, structure, and invariants
should look identical. What changes is the backend and how long you run it.

## Quickstart (libfuzzer_sys)

```rust
use chaos_theory::fuzz_target_libfuzzer_sys;

fuzz_target_libfuzzer_sys!(|src| {
    let v: Vec<u8> = src.any("v");
    // your invariants here
});
```

## Seeds

Before fuzzing, generate seeds:

```rust
use chaos_theory::fuzz_write_seed;

let _ = fuzz_write_seed("corpus", |src| {
    let _v: Vec<u8> = src.any("v");
});
```

Good defaults:

- Start with ~32 seeds for simple systems.
- Increase for complex systems.
- Minimize the corpus before long fuzzing runs.

## Direct Hooks (Advanced)

If you need custom integration:

- `fuzz_check` validates fuzzer-provided input.
- `fuzz_mutate` mutates input with structure-aware rules.
- `fuzz_mutate_crossover` combines two inputs.

These are the same building blocks used by the `libfuzzer_sys` wrapper.

## Notes

- Properties should be written exactly like property tests.
- Avoid external randomness and wall-clock dependencies inside the property.
- Use `CHAOS_THEORY_REPLAY=...` to reproduce a failure found by the fuzzer.
