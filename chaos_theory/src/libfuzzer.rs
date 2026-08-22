// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// Define a `libfuzzer_sys` fuzz target compatible with `cargo-fuzz`.
///
/// This requires the [`chaos_theory_libfuzzer`](https://crates.io/crates/chaos_theory_libfuzzer)
/// fork as the package for the `libfuzzer-sys` dependency.
///
/// Example:
///
/// ```rust
/// use chaos_theory::{fuzz_target_libfuzzer, Source};
///
/// fn prop(src: &mut Source) {
///     let points: Vec<(i32, i32)> = src.any("points");
///     // invariants here
/// }
///
/// fuzz_target_libfuzzer!(prop);
/// ```
///
/// Fuzzer failures are not minimized to avoid triggering libFuzzer timeout detection;
/// replay with `CHAOS_THEORY_REPLAY=...` using [`check`](crate::check) to reproduce and minimize.
#[macro_export]
macro_rules! fuzz_target_libfuzzer {
    ($prop:expr) => {
        ::std::thread_local! {
            static _CHAOS_THEORY_FUZZ_STATE: ::std::cell::RefCell<$crate::__private::FuzzState> =
                ::std::cell::RefCell::new($crate::__private::FuzzState::new());
        }

        static _CHAOS_THEORY_EFFECTIVE_INPUT_SUPPORTED: ::std::sync::atomic::AtomicBool =
            ::std::sync::atomic::AtomicBool::new(false);

        ::libfuzzer_sys::fuzz_mutator!(|data: &mut [u8],
                                        size: usize,
                                        max_size: usize,
                                        seed: u32| {
            _CHAOS_THEORY_FUZZ_STATE.with_borrow_mut(|state| {
                $crate::Env::builder()
                    .with_env_vars()
                    .build()
                    .fuzz_mutate(state, data, size, max_size, seed, true, None)
            })
        });

        ::libfuzzer_sys::fuzz_crossover!(|input: &[u8],
                                          other: &[u8],
                                          out: &mut [u8],
                                          seed: u32| {
            _CHAOS_THEORY_FUZZ_STATE.with_borrow_mut(|state| {
                $crate::Env::builder()
                    .with_env_vars()
                    .build()
                    .fuzz_mutate_crossover(state, input, other, out, seed, true)
            })
        });

        ::libfuzzer_sys::fuzz_target!(
            init: {
                assert!(
                    _CHAOS_THEORY_EFFECTIVE_INPUT_SUPPORTED
                        .load(::std::sync::atomic::Ordering::SeqCst),
                    "chaos_theory fuzz targets require effective-input support; use \
                     package = \"chaos_theory_libfuzzer\" for the libfuzzer-sys dependency"
                );
            },
            |input: &[u8]| -> ::libfuzzer_sys::Corpus {
                let interesting = _CHAOS_THEORY_FUZZ_STATE.with_borrow_mut(|state| {
                    $crate::Env::builder()
                        .with_env_vars()
                        .build()
                        .fuzz_check(state, input, $prop)
                        .is_some()
                });
                if interesting {
                    ::libfuzzer_sys::Corpus::Keep
                } else {
                    ::libfuzzer_sys::Corpus::Reject
                }
            }
        );

        #[unsafe(no_mangle)]
        pub extern "C" fn LLVMFuzzerRequireEffectiveInput() {
            _CHAOS_THEORY_EFFECTIVE_INPUT_SUPPORTED
                .store(true, ::std::sync::atomic::Ordering::SeqCst);
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn LLVMFuzzerCustomGetEffectiveInput(
            _data: *const u8,
            _size: usize,
            out: *mut u8,
            max_out_size: usize,
        ) -> usize {
            _CHAOS_THEORY_FUZZ_STATE.with_borrow(|state| {
                let effective = state.effective_input().expect(
                    "internal error: effective input must follow an accepted fuzz execution",
                );
                if effective.len() <= max_out_size {
                    assert!(!out.is_null());
                    unsafe {
                        ::core::ptr::copy_nonoverlapping(
                            effective.as_ptr(),
                            out,
                            effective.len(),
                        );
                    }
                }
                effective.len()
            })
        }
    };
}
