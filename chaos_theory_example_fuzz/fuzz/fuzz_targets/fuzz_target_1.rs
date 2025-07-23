#![no_main]

chaos_theory::fuzz_target_libfuzzer_sys!(chaos_theory_example_fuzz::prop_fuzz_target_1);
