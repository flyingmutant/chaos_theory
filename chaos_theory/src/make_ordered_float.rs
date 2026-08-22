// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{Arbitrary, Float, Generator, MaybeOwned, make_float::float_in};
use core::fmt::Debug;
use core::ops::RangeBounds;
use ordered_float::{FloatCore, NotNan, OrderedFloat};

#[cfg_attr(docsrs, doc(cfg(feature = "ordered_float")))]
impl<F> Arbitrary for OrderedFloat<F>
where
    F: Arbitrary + Float + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        ordered_float(F::arbitrary())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ordered_float")))]
impl<F> Arbitrary for NotNan<F>
where
    F: Float + Debug + FloatCore,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        not_nan_in(..)
    }
}

/// Create an [`OrderedFloat`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "ordered_float")))]
pub fn ordered_float<F>(f: impl Generator<Item = F>) -> impl Generator<Item = OrderedFloat<F>>
where
    F: Float + Debug,
{
    f.map_reversible(OrderedFloat, |f| Some(MaybeOwned::Owned(f.0)))
}

/// Create an [`OrderedFloat`] generator constrained by `range`.
#[cfg_attr(docsrs, doc(cfg(feature = "ordered_float")))]
pub fn ordered_float_in<F>(range: impl RangeBounds<F>) -> impl Generator<Item = OrderedFloat<F>>
where
    F: Float + Debug,
{
    ordered_float(float_in::<F>(range))
}

/// Create a [`NotNan`] generator from a float generator.
///
/// NaNs produced by `f` are treated as failed assumptions and discarded.
#[cfg_attr(docsrs, doc(cfg(feature = "ordered_float")))]
#[expect(clippy::missing_panics_doc)]
pub fn not_nan<F>(f: impl Generator<Item = F>) -> impl Generator<Item = NotNan<F>>
where
    F: Float + Debug + FloatCore,
{
    f.filter(|f| !f.is_nan()).map_reversible(
        |f| NotNan::new(f).expect("internal error: NotNan::new failed after filter"),
        |n| Some(MaybeOwned::Owned(n.into_inner())),
    )
}

/// Create a [`NotNan`] generator constrained by `range`.
#[cfg_attr(docsrs, doc(cfg(feature = "ordered_float")))]
#[expect(clippy::missing_panics_doc)]
pub fn not_nan_in<F>(range: impl RangeBounds<F>) -> impl Generator<Item = NotNan<F>>
where
    F: Float + Debug + FloatCore,
{
    float_in::<F>(range).map_reversible(
        |f| NotNan::new(f).expect("internal error: float_in generated NaN"),
        |n| Some(MaybeOwned::Owned(n.into_inner())),
    )
}

#[cfg(test)]
mod tests {
    use crate::{check, make, tests::prop_smoke};

    #[test]
    fn ordered_float_smoke() {
        check(|src| {
            prop_smoke(
                src,
                "OrderedFloat<f32>",
                make::ordered_float::ordered_float_in::<f32>(..),
            );
            prop_smoke(
                src,
                "OrderedFloat<f64>",
                make::ordered_float::ordered_float_in::<f64>(..),
            );
            prop_smoke(
                src,
                "NotNan<f32>",
                make::ordered_float::not_nan_in::<f32>(..),
            );
            prop_smoke(
                src,
                "NotNan<f64>",
                make::ordered_float::not_nan_in::<f64>(..),
            );
        });
    }
}
