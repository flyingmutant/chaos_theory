// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::make_string::{CharBuf, next_string_impl};
use crate::{Arbitrary, Effect, Generator, SourceEx, make, range::SizeRange};
use core::{fmt::Debug, marker::PhantomData, ops::RangeBounds};
use ecow::{EcoBytes, EcoString, EcoVec};

#[cfg_attr(docsrs, doc(cfg(feature = "ecow")))]
impl Arbitrary for EcoBytes {
    fn arbitrary() -> impl Generator<Item = Self> {
        eco_bytes(<u8 as Arbitrary>::arbitrary())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ecow")))]
impl Arbitrary for EcoString {
    fn arbitrary() -> impl Generator<Item = Self> {
        eco_string(<char as Arbitrary>::arbitrary())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ecow")))]
impl<T> Arbitrary for EcoVec<T>
where
    T: Arbitrary + Clone + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        eco_vec(T::arbitrary())
    }
}

/// Create an [`EcoBytes`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "ecow")))]
pub fn eco_bytes(elem: impl Generator<Item = u8>) -> impl Generator<Item = EcoBytes> {
    eco_bytes_n(elem, ..)
}

/// Create an [`EcoBytes`] generator with the specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "ecow")))]
pub fn eco_bytes_n(
    elem: impl Generator<Item = u8>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = EcoBytes> {
    make::slice_n(elem, size)
}

impl CharBuf for EcoString {
    fn reserve(&mut self, size: usize) {
        if self.is_empty() {
            *self = Self::with_capacity(size);
        }
    }

    fn push_char(&mut self, ch: char) {
        self.push(ch);
    }
}

#[derive(Debug)]
struct EcoString_<G> {
    elem: G,
    size: SizeRange,
}

impl<G> Generator for EcoString_<G>
where
    G: Generator<Item = char>,
{
    type Item = EcoString;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let mut s = EcoString::new();
        next_string_impl(
            src,
            example.map(|e| e.as_str().as_bytes()),
            &mut s,
            &self.elem,
            self.size,
        );
        s
    }
}

/// Create an [`EcoString`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "ecow")))]
pub fn eco_string(elem: impl Generator<Item = char>) -> impl Generator<Item = EcoString> {
    eco_string_n(elem, ..)
}

/// Create an [`EcoString`] generator with the specified size (in characters).
#[cfg_attr(docsrs, doc(cfg(feature = "ecow")))]
pub fn eco_string_n(
    elem: impl Generator<Item = char>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = EcoString> {
    EcoString_ {
        elem,
        size: SizeRange::new(size),
    }
}

#[derive(Debug)]
struct EcoVec_<G, T> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<T>,
}

impl<G, T> Generator for EcoVec_<G, T>
where
    G: Generator<Item = T>,
    T: Clone + Debug,
{
    type Item = EcoVec<T>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.as_slice().iter());
        let res = src.repeat(
            "<ecovec>",
            example_seq,
            self.size,
            |n| EcoVec::with_capacity(n),
            |v, src, example| {
                let elem = self.elem.next(src, example);
                v.push(elem);
                Effect::Success
            },
        );
        res.expect("internal error: ecovec element repeat should not fail")
    }
}

/// Create an [`EcoVec`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "ecow")))]
pub fn eco_vec<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = EcoVec<T>>
where
    T: Clone + Debug,
{
    eco_vec_n(elem, ..)
}

/// Create an [`EcoVec`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "ecow")))]
pub fn eco_vec_n<T>(
    elem: impl Generator<Item = T>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = EcoVec<T>>
where
    T: Clone + Debug,
{
    EcoVec_ {
        elem,
        size: SizeRange::new(size),
        _marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use crate::{check, make, tests::prop_smoke};

    #[test]
    fn ecow_smoke() {
        const MAX_CHARS: usize = 12;
        const MAX_SIZE: usize = 8;

        check(|src| {
            prop_smoke(
                src,
                "EcoBytes",
                make::ecow::eco_bytes_n(make::arbitrary::<u8>(), ..MAX_SIZE),
            );
            prop_smoke(
                src,
                "EcoString",
                make::ecow::eco_string_n(make::arbitrary::<char>(), ..MAX_CHARS),
            );
            prop_smoke(
                src,
                "EcoVec<i32>",
                make::ecow::eco_vec_n(make::arbitrary::<i32>(), ..MAX_SIZE),
            );
        });
    }
}
