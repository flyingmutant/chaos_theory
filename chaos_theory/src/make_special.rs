// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::ops::RangeBounds;

use crate::{Generator, SourceEx, Tweak, range::SizeRange};

#[derive(Debug)]
struct Size {
    r: SizeRange,
}

impl Generator for Size {
    type Item = usize;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        src.as_mut().choose_size(self.r, example.copied())
    }
}

/// Create a generator of valid size values.
///
/// When possible, prefer using [`Source::repeat`][crate::Source::repeat]
/// or [`SourceEx::repeat`][crate::SourceEx::repeat].
///
/// # Panics
///
/// `size` panics when the range is invalid or empty.
pub fn size(r: impl RangeBounds<usize>) -> impl Generator<Item = usize> {
    Size {
        r: SizeRange::new(r),
    }
}

#[derive(Debug)]
struct Index {
    n: usize,
}

impl Generator for Index {
    type Item = usize;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        src.as_mut()
            .choose_index(self.n, example.copied(), Tweak::None)
    }
}

/// Create a generator of valid index values.
///
/// When possible, prefer using [`Source::choose`][crate::Source::choose] and
/// [`Source::select`][crate::Source::select].
///
/// # Panics
///
/// `index` panics when `n` is zero.
#[track_caller]
pub fn index(n: usize) -> impl Generator<Item = usize> {
    assert!(n != 0, "index range is empty");
    Index { n }
}

#[derive(Debug)]
struct TryIndex {
    n: usize,
}

impl Generator for TryIndex {
    type Item = Option<usize>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        (self.n != 0).then(|| {
            src.as_mut()
                .choose_index(self.n, example.copied().flatten(), Tweak::None)
        })
    }
}

/// Create a generator that tries to produce valid index values.
///
/// The generator yields `None` when `n` is zero.
///
/// When possible, prefer using [`Source::choose`][crate::Source::choose] and
/// [`Source::select`][crate::Source::select].
pub fn try_index(n: usize) -> impl Generator<Item = Option<usize>> {
    TryIndex { n }
}

#[derive(Debug)]
struct Token;

impl Generator for Token {
    type Item = u128;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_lo = example.map(|v| *v as u64);
        let example_hi = example.map(|v| (v >> 64) as u64);
        let lo = src.as_mut().choose_token(example_lo);
        let hi = src.as_mut().choose_token(example_hi);
        (u128::from(hi) << 64) | u128::from(lo)
    }
}

/// Create a generator of random unique token values.
pub fn token() -> impl Generator<Item = u128> {
    Token
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        check,
        tests::{print_debug_examples, prop_smoke},
    };

    #[test]
    fn size_smoke() {
        check(|src| {
            prop_smoke(src, "size", size(..));
        });
    }

    #[test]
    fn index_smoke() {
        check(|src| {
            prop_smoke(src, "index", index(100));
            let n = src.any("n");
            prop_smoke(src, "try_index", try_index(n));
        });
    }

    #[test]
    fn size_examples() {
        print_debug_examples(size(..), None, Ord::cmp);
    }

    #[test]
    fn index_examples() {
        const N: usize = 5;
        const M: usize = 100;
        let g = index(N).collect_n::<Vec<_>>(M..=M).map(|v| {
            let mut u: [usize; N] = [0; N];
            for i in v {
                u[i] += 1;
            }
            u
        });
        print_debug_examples(g, None, Ord::cmp);
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use crate::{make, tests::bench_gen_next};
    use core::hint::black_box;

    #[bench]
    fn gen_size(b: &mut test::Bencher) {
        let g = make::size(black_box(..));
        bench_gen_next(b, &g);
    }

    #[bench]
    fn gen_index(b: &mut test::Bencher) {
        let g = make::index(black_box(100));
        bench_gen_next(b, &g);
    }
}
