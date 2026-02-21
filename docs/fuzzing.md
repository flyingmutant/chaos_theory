# Fuzzing

The central idea in chaos_theory is that property testing and fuzzing are the
same thing at different time scales. The properties, structure, and invariants
should look identical. What changes is the backend and how long you run it.

## Quickstart (libfuzzer_sys)

```rust
use chaos_theory::{fuzz_target_libfuzzer_sys, Source};

fn prop(src: &mut Source) {
    let points: Vec<(i32, i32)> = src.any("points");
    // your invariants here
}

fuzz_target_libfuzzer_sys!(prop);
```

Fuzzer failures are non-minimized to avoid triggering a fuzzer timeout during minimization.
A common pattern is to use the same property in a regular `check` test and run it with
the `CHAOS_THEORY_REPLAY=...` value found during a fuzzing run to minimize and reproduce
a failure without using the fuzzer.

## Seeds

Before fuzzing, generate the seeds once:

```rust
use chaos_theory::fuzz_write_seed;

for _ in 0..32 {
    fuzz_write_seed("corpus", prop).unwrap();
}
```

This can conveniently be done from an ignored test.

## Direct Hooks (Advanced)

If you need custom fuzzer integration:

- `fuzz_check` validates fuzzer-provided input.
- `fuzz_mutate` mutates input with structure-aware rules.
- `fuzz_mutate_crossover` combines two inputs.

These are the same building blocks used by the `libfuzzer_sys` wrapper.
