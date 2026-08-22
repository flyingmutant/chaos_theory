// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{fmt::Debug, num::NonZero};

use either::Either;

use crate::{Arbitrary, Generator, MaybeOwned, SourceEx};

#[cfg_attr(docsrs, doc(cfg(feature = "either")))]
impl<L: Arbitrary, R: Arbitrary> Arbitrary for Either<L, R> {
    fn arbitrary() -> impl Generator<Item = Self> {
        either(L::arbitrary(), R::arbitrary())
    }
}

#[derive(Debug)]
struct Either_<GL, GR> {
    left: GL,
    right: GR,
}

impl<GL: Generator, GR: Generator> Generator for Either_<GL, GR> {
    type Item = Either<GL::Item, GR::Item>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_index = example.map(|e| match e {
            Either::Left(_) => 0,
            Either::Right(_) => 1,
        });

        let variants = &["Left", "Right"];
        let variants_num = NonZero::new(variants.len()).expect("internal error: no variants");
        src.select(
            "<either>",
            example_index,
            variants_num,
            |ix| variants[ix],
            |src, variant, _ix| match variant {
                "Left" => {
                    let example_left = match example {
                        Some(Either::Left(left)) => Some(left),
                        _ => None,
                    };
                    Either::Left(self.left.next(src, example_left))
                }
                "Right" => {
                    let example_right = match example {
                        Some(Either::Right(right)) => Some(right),
                        _ => None,
                    };
                    Either::Right(self.right.next(src, example_right))
                }
                _ => unreachable!(),
            },
        )
    }
}

/// Create an [`Either`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "either")))]
pub fn either<L: Debug, R: Debug>(
    left: impl Generator<Item = L>,
    right: impl Generator<Item = R>,
) -> impl Generator<Item = Either<L, R>> {
    Either_ { left, right }
}

/// Create an [`Either`] generator that always generates [`Either::Left`] values.
#[cfg_attr(docsrs, doc(cfg(feature = "either")))]
pub fn left<L: Debug, R: Debug>(
    g: impl Generator<Item = L>,
) -> impl Generator<Item = Either<L, R>> {
    g.map_reversible(Either::Left, |e| match e {
        Either::Left(left) => Some(MaybeOwned::Borrowed(left)),
        Either::Right(_) => None,
    })
}

/// Create an [`Either`] generator that always generates [`Either::Right`] values.
#[cfg_attr(docsrs, doc(cfg(feature = "either")))]
pub fn right<L: Debug, R: Debug>(
    g: impl Generator<Item = R>,
) -> impl Generator<Item = Either<L, R>> {
    g.map_reversible(Either::Right, |e| match e {
        Either::Left(_) => None,
        Either::Right(right) => Some(MaybeOwned::Borrowed(right)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, make, tests::prop_smoke};

    #[test]
    fn either_smoke() {
        check(|src| {
            prop_smoke(
                src,
                "either arbitrary",
                make::arbitrary::<Either<i8, i32>>(),
            );
            prop_smoke(src, "either", either(i8::arbitrary(), i32::arbitrary()));
            prop_smoke(src, "left", left::<_, i32>(i8::arbitrary()));
            prop_smoke(src, "right", right::<i8, _>(i32::arbitrary()));
        });
    }
}
