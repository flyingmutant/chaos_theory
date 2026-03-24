// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::ops::RangeBounds;

use bytes::{BufMut as _, Bytes, BytesMut};

use crate::{Arbitrary, Effect, Generator, SourceRaw, make, range::SizeRange};

#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
impl Arbitrary for Bytes {
    fn arbitrary() -> impl Generator<Item = Self> {
        bytes(make::arbitrary())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
impl Arbitrary for BytesMut {
    fn arbitrary() -> impl Generator<Item = Self> {
        bytes_mut(make::arbitrary())
    }
}

/// Create a [`Bytes`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
pub fn bytes(elem: impl Generator<Item = u8>) -> impl Generator<Item = Bytes> {
    bytes_with_size(elem, ..)
}

/// Create a [`Bytes`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
pub fn bytes_with_size(
    elem: impl Generator<Item = u8>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = Bytes> {
    make::slice_with_size(elem, size)
}

#[derive(Debug)]
struct BytesMut_<G> {
    elem: G,
    size: SizeRange,
}

impl<G: Generator<Item = u8>> Generator for BytesMut_<G> {
    type Item = BytesMut;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<bytes_mut>",
            example_seq,
            self.size,
            BytesMut::with_capacity,
            |v, src, example| {
                let byte = self.elem.next(src, example);
                v.put_u8(byte);
                Effect::Success
            },
        );
        res.expect("internal error: bytes_mut repeat should not fail")
    }
}

/// Create a [`BytesMut`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
pub fn bytes_mut(elem: impl Generator<Item = u8>) -> impl Generator<Item = BytesMut> {
    bytes_mut_with_size(elem, ..)
}

/// Create a [`BytesMut`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
pub fn bytes_mut_with_size(
    elem: impl Generator<Item = u8>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = BytesMut> {
    BytesMut_ {
        elem,
        size: SizeRange::new(size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, make, tests::prop_smoke};

    #[test]
    fn bytes_smoke() {
        check(|src| {
            prop_smoke(src, "Bytes", bytes(make::arbitrary()));
            prop_smoke(src, "BytesMut", bytes_mut(make::arbitrary()));
        });
    }
}
