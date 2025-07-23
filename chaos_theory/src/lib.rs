// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/*!
`chaos_theory` is a modern property-based testing and structure-aware fuzzing library.
*/

#![cfg_attr(all(test, feature = "_bench"), feature(test))]

extern crate alloc;
#[cfg(all(test, feature = "_bench"))]
extern crate test;

use std::path::Path;

mod base64;
mod config;
mod cover;
mod distrib;
mod env;
mod generator;
mod hash;
mod hash_identity;
mod libfuzzer;
mod math;
mod num;
mod permute;
mod rand;
mod range;
mod reduce;
mod source;
mod tape;
mod tape_event;
mod tape_mutate;
mod tape_mutate_crossover;
mod tape_reduce;
mod tape_validate;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_shrinking_challenge;
mod unwind;
mod util;
mod varint;

mod make_cell;
mod make_char;
mod make_collection;
mod make_combine;
mod make_core;
mod make_float;
mod make_integer;
#[cfg(feature = "regex")]
mod make_regex;
mod make_special;
mod make_string;
mod make_sync;
mod make_time;
mod make_tuple;

pub use config::*;
pub use env::*;
pub use generator::*;
pub use num::*;
pub use source::*;
pub use unwind::*;
pub use util::*;

/// Additional documentation.
pub mod _docs {
    #[doc = include_str!("../../docs/FAQ.md")]
    pub mod faq {}
    #[doc = include_str!("../../docs/CHANGELOG.md")]
    pub mod changelog {}
}

/// Collection of built-in generator implementations.
pub mod make {
    pub use crate::make_cell::*;
    pub use crate::make_char::*;
    pub use crate::make_collection::*;
    pub use crate::make_combine::*;
    pub use crate::make_core::*;
    pub use crate::make_float::*;
    pub use crate::make_integer::*;
    #[cfg(feature = "regex")]
    pub use crate::make_regex::*;
    pub use crate::make_special::*;
    pub use crate::make_string::*;
    pub use crate::make_sync::*;
    pub use crate::make_time::*;

    use crate::{Arbitrary, Generator};

    /// Create a generator of `T`, using its [`Arbitrary`] implementation.
    ///
    /// This is equivalent to `<T as Arbitrary>::arbitrary()`, but can sometimes be a bit more concise.
    pub fn arbitrary<T: Arbitrary>() -> impl Generator<Item = T> {
        T::arbitrary()
    }
}

/// Check that property holds (does not panic).
///
/// To customize the `check` behavior, use [`Env::custom`] and [`Env::check`]
/// (or specify environment variables mentioned in [`Config::env`](crate::Config::env) documentation).
pub fn check(prop: impl Fn(&mut Source)) {
    Env::new().check(prop);
}

/// Write seed input for the fuzzer. This can conveniently be done from an ignored test.
///
/// A good default number of seeds to write before starting fuzzing is 32.
/// Fewer seeds are required for simple systems, while complex ones can benefit from
/// more seeds (at the expense of slowing down fuzzing due to seed redundancy).
/// Recommended approach is to write a lot of seeds (e.g. 1024), and apply
/// corpus minimization to them before starting fuzzing.
///
/// To customize the `write_fuzz_seed` behavior, use [`Env::custom`] and [`Env::fuzz_write_seed`]
/// (or specify environment variables mentioned in [`Config::env`](crate::Config::env) documentation).
///
/// # Errors
///
/// `fuzz_write_seed` fails when valid test case can not be generated or in case of a filesystem error.
pub fn fuzz_write_seed(
    seed_dir: impl AsRef<Path>,
    prop: impl Fn(&mut Source),
) -> Result<(), &'static str> {
    Env::new().fuzz_write_seed(seed_dir, prop)
}

/// Check that property holds (does not panic) on fuzzer-provided input.
///
/// You probably want to use higher-level wrapper like
/// [`fuzz_target_libfuzzer_sys`] instead of manually invoking this function.
#[must_use]
pub fn fuzz_check(
    input: &[u8],
    out: Option<(&mut [u8], &mut usize)>,
    prop: impl Fn(&mut Source),
) -> bool {
    Env::new().fuzz_check(input, out, prop)
}

/// Mutate fuzzer input.
///
/// You probably want to use higher-level wrapper like
/// [`fuzz_target_libfuzzer_sys`] instead of manually invoking this function.
#[expect(clippy::type_complexity)]
pub fn fuzz_mutate(
    data: &mut [u8],
    size: usize,
    max_size: usize,
    seed: u32,
    allow_void: bool,
    mutate_bin: Option<fn(&mut [u8], usize, usize) -> usize>,
) -> usize {
    Env::new().fuzz_mutate(data, size, max_size, seed, allow_void, mutate_bin)
}

/// Cross-over two fuzzer inputs.
///
/// You probably want to use higher-level wrapper like
/// [`fuzz_target_libfuzzer_sys`] instead of manually invoking this function.
pub fn fuzz_mutate_crossover(
    input: &[u8],
    other: &[u8],
    out: &mut [u8],
    seed: u32,
    allow_void: bool,
) -> usize {
    Env::new().fuzz_mutate_crossover(input, other, out, seed, allow_void)
}

pub(crate) type Set<K> = std::collections::HashSet<K, hash::FxBuildHasher>;
pub(crate) type Map<K, V> = std::collections::HashMap<K, V, hash::FxBuildHasher>;
