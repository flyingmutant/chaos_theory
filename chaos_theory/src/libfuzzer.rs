// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// Define a [`libfuzzer_sys`](https://docs.rs/libfuzzer-sys) fuzz target.
///
/// Don't forget to use [`crate::fuzz_write_seed`] before fuzzing.
///
/// Example:
///
/// ```rust
/// use chaos_theory::{fuzz_target_libfuzzer_sys, Source};
///
/// fn prop(src: &mut Source) {
///     let points: Vec<(i32, i32)> = src.any("points");
///     // invariants here
/// }
///
/// fuzz_target_libfuzzer_sys!(prop);
/// ```
///
/// Fuzzer failures are not minimized to avoid triggering libFuzzer timeout detection;
/// replay with `CHAOS_THEORY_REPLAY=...` using [`crate::check`] to reproduce and minimize.
#[macro_export]
macro_rules! fuzz_target_libfuzzer_sys {
    ($prop:expr) => {
        ::libfuzzer_sys::fuzz_mutator!(|data: &mut [u8],
                                        size: usize,
                                        max_size: usize,
                                        seed: u32| {
            $crate::fuzz_mutate(data, size, max_size, seed, false, None)
        });

        ::libfuzzer_sys::fuzz_crossover!(|input: &[u8],
                                          other: &[u8],
                                          out: &mut [u8],
                                          seed: u32| {
            $crate::fuzz_mutate_crossover(input, other, out, seed, false)
        });

        ::libfuzzer_sys::fuzz_target!(|input: &[u8]| -> ::libfuzzer_sys::Corpus {
            let interesting = $crate::fuzz_check(input, None, $prop);
            if interesting {
                ::libfuzzer_sys::Corpus::Keep
            } else {
                ::libfuzzer_sys::Corpus::Reject
            }
        });
    };
}
