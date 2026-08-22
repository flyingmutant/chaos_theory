// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    ops::RangeBounds,
};
use tinyvec::{Array, ArrayVec, TinyVec};

use crate::{Arbitrary, Effect, Generator, SourceEx, range::SizeRange};

#[cfg_attr(docsrs, doc(cfg(feature = "tinyvec")))]
impl<A> Arbitrary for TinyVec<A>
where
    A: Array,
    A::Item: Arbitrary + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        tiny_vec::<A>(<A::Item as Arbitrary>::arbitrary())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "tinyvec")))]
impl<A> Arbitrary for ArrayVec<A>
where
    A: Array,
    A::Item: Arbitrary + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        array_vec::<A>(<A::Item as Arbitrary>::arbitrary())
    }
}

struct TinyVec_<G, A> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<A>,
}

impl<G: Debug, A> Debug for TinyVec_<G, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TinyVec_")
            .field("elem", &self.elem)
            .field("size", &self.size)
            .finish()
    }
}

impl<G, A> Generator for TinyVec_<G, A>
where
    G: Generator<Item = A::Item>,
    A: Array,
    A::Item: Debug,
{
    type Item = TinyVec<A>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<tinyvec>",
            example_seq,
            self.size,
            |n| TinyVec::<A>::with_capacity(n),
            |v, src, example| {
                let elem = self.elem.next(src, example);
                v.push(elem);
                Effect::Success
            },
        );
        res.expect("internal error: tinyvec repeat should not fail")
    }
}

struct ArrayVec_<G, A> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<A>,
}

impl<G: Debug, A> Debug for ArrayVec_<G, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrayVec_")
            .field("elem", &self.elem)
            .field("size", &self.size)
            .finish()
    }
}

impl<G, A> Generator for ArrayVec_<G, A>
where
    G: Generator<Item = A::Item>,
    A: Array,
    A::Item: Debug,
{
    type Item = ArrayVec<A>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<arrayvec>",
            example_seq,
            self.size,
            |_n| ArrayVec::<A>::default(),
            |v, src, example| {
                let elem = self.elem.next(src, example);
                let overflow = v.try_push(elem);
                assert!(
                    overflow.is_none(),
                    "internal error: arrayvec capacity exceeded during generation"
                );
                Effect::Success
            },
        );
        res.expect("internal error: arrayvec repeat should not fail")
    }
}

/// Create a [`TinyVec`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "tinyvec")))]
pub fn tiny_vec<A>(elem: impl Generator<Item = A::Item>) -> impl Generator<Item = TinyVec<A>>
where
    A: Array,
    A::Item: Debug,
{
    tiny_vec_with_size(elem, ..)
}

/// Create a [`TinyVec`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "tinyvec")))]
pub fn tiny_vec_with_size<A>(
    elem: impl Generator<Item = A::Item>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = TinyVec<A>>
where
    A: Array,
    A::Item: Debug,
{
    let size = SizeRange::new(size);
    TinyVec_ {
        elem,
        size,
        _marker: PhantomData,
    }
}

/// Create an [`ArrayVec`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "tinyvec")))]
pub fn array_vec<A>(elem: impl Generator<Item = A::Item>) -> impl Generator<Item = ArrayVec<A>>
where
    A: Array,
    A::Item: Debug,
{
    array_vec_with_size(elem, 0..=A::CAPACITY)
}

/// Create an [`ArrayVec`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "tinyvec")))]
#[expect(clippy::missing_panics_doc)]
pub fn array_vec_with_size<A>(
    elem: impl Generator<Item = A::Item>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = ArrayVec<A>>
where
    A: Array,
    A::Item: Debug,
{
    let size = SizeRange::new(size);
    assert!(
        size.max <= A::CAPACITY,
        "array_vec_with_size upper bound {} exceeds capacity {}",
        size.max,
        A::CAPACITY
    );
    ArrayVec_ {
        elem,
        size,
        _marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use crate::{check, make, tests::prop_smoke};

    #[test]
    fn tinyvec_smoke() {
        const MAX_INLINE: usize = 4;
        const MAX_SMALL: usize = 8;

        check(|src| {
            prop_smoke(
                src,
                "ArrayVec<[i32; 4]>",
                make::tinyvec::array_vec_with_size::<[i32; MAX_INLINE]>(
                    make::arbitrary::<i32>(),
                    0..=MAX_INLINE,
                ),
            );
            prop_smoke(
                src,
                "TinyVec<[i32; 4]>",
                make::tinyvec::tiny_vec_with_size::<[i32; MAX_INLINE]>(
                    make::arbitrary::<i32>(),
                    0..=MAX_SMALL,
                ),
            );
        });
    }
}
