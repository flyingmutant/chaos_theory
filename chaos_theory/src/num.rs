// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![expect(clippy::allow_attributes)]

use core::{
    fmt::{Debug, Display},
    hash::Hash,
    ops::{Add, Shr, Sub},
};

use crate::math::bitmask;

#[doc(hidden)]
pub trait Ranged: 'static + Debug + Copy + PartialOrd {
    const ZERO: Self;
    const MIN: Self;
    const MAX: Self;

    fn next_up(self) -> Option<Self>;
    fn next_down(self) -> Option<Self>;
}

#[doc(hidden)]
pub trait Int:
    'static
    + Copy
    + Eq
    + Ord
    + Hash
    + Display
    + Debug
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Shr<usize, Output = Self>
    + Ranged
{
    const UNSIGNED: bool;
    type Unsigned: Unsigned;

    #[must_use]
    fn is_negative(self) -> bool;

    #[must_use]
    fn unsigned_abs(self) -> Self::Unsigned;

    #[must_use]
    fn from_unsigned(u: Self::Unsigned) -> Self;

    #[must_use]
    fn wrapping_neg(self) -> Self;
}

#[doc(hidden)]
pub trait Unsigned: Int {
    const BITS: usize;

    #[must_use]
    fn bit_len(self) -> usize;

    #[must_use]
    fn to_bits(self) -> u64;

    #[must_use]
    fn from_bits(w: u64) -> Self;
}

macro_rules! impl_int_unsigned {
    ($num: ty) => {
        impl Ranged for $num {
            const ZERO: Self = 0;
            const MIN: Self = <$num>::MIN;
            const MAX: Self = <$num>::MAX;

            fn next_up(self) -> Option<Self> {
                self.checked_add(1)
            }
            fn next_down(self) -> Option<Self> {
                self.checked_sub(1)
            }
        }

        impl Int for $num {
            const UNSIGNED: bool = true;
            type Unsigned = Self;

            fn is_negative(self) -> bool {
                false
            }
            fn unsigned_abs(self) -> Self::Unsigned {
                self
            }
            fn from_unsigned(u: Self::Unsigned) -> Self {
                u
            }
            fn wrapping_neg(self) -> Self {
                self.wrapping_neg()
            }
        }

        impl Unsigned for $num {
            const BITS: usize = <$num>::BITS as usize;

            fn bit_len(self) -> usize {
                <Self as Unsigned>::BITS - (self.leading_zeros() as usize)
            }
            #[allow(clippy::cast_lossless)]
            fn to_bits(self) -> u64 {
                self as u64
            }
            #[allow(clippy::cast_lossless)]
            fn from_bits(w: u64) -> Self {
                w as Self
            }
        }
    };
}

macro_rules! impl_int_signed {
    ($num: ty, $unsigned: ty) => {
        impl Ranged for $num {
            const ZERO: Self = 0;
            const MIN: Self = <$num>::MIN;
            const MAX: Self = <$num>::MAX;

            fn next_up(self) -> Option<Self> {
                self.checked_add(1)
            }
            fn next_down(self) -> Option<Self> {
                self.checked_sub(1)
            }
        }

        impl Int for $num {
            const UNSIGNED: bool = false;
            type Unsigned = $unsigned;

            fn is_negative(self) -> bool {
                self.is_negative()
            }
            fn unsigned_abs(self) -> Self::Unsigned {
                self.unsigned_abs()
            }
            fn from_unsigned(u: Self::Unsigned) -> Self {
                #[allow(clippy::cast_possible_wrap)]
                return u as Self;
            }
            fn wrapping_neg(self) -> Self {
                self.wrapping_neg()
            }
        }
    };
}

impl_int_unsigned!(usize);
impl_int_unsigned!(u8);
impl_int_unsigned!(u16);
impl_int_unsigned!(u32);
impl_int_unsigned!(u64);

impl_int_signed!(isize, usize);
impl_int_signed!(i8, u8);
impl_int_signed!(i16, u16);
impl_int_signed!(i32, u32);
impl_int_signed!(i64, u64);

#[doc(hidden)]
pub trait Float:
    'static + Copy + Display + Debug + Add<Self, Output = Self> + Sub<Self, Output = Self> + Ranged
{
    const MANTISSA_BITS: u64;
    const EXPONENT_BIAS: i32;

    #[must_use]
    fn from_bits(u: u64) -> Self;

    #[must_use]
    fn to_bits_unsigned(self) -> u64;

    #[must_use]
    fn is_negative(self) -> bool;

    #[must_use]
    fn negate(self) -> Self;
}

macro_rules! impl_float {
    ($num : ty, $u : ty) => {
        impl Ranged for $num {
            const ZERO: Self = 0.0;
            const MIN: Self = <$num>::NEG_INFINITY;
            const MAX: Self = <$num>::INFINITY;

            fn next_up(self) -> Option<Self> {
                Some(self.next_up())
            }

            fn next_down(self) -> Option<Self> {
                Some(self.next_down())
            }
        }

        impl Float for $num {
            const MANTISSA_BITS: u64 = Self::MANTISSA_DIGITS as u64 - 1;
            const EXPONENT_BIAS: i32 =
                (1 << ((size_of::<Self>() * 8) as u64 - Self::MANTISSA_BITS - 2)) - 1;

            fn from_bits(u: u64) -> Self {
                Self::from_bits(u as $u)
            }

            fn to_bits_unsigned(self) -> u64 {
                u64::from(self.to_bits()) & bitmask::<u64>((size_of::<Self>() * 8) - 1)
            }

            fn is_negative(self) -> bool {
                self < Self::ZERO
            }

            fn negate(self) -> Self {
                -self
            }
        }
    };
}

impl_float!(f32, u32);
impl_float!(f64, u64);

#[cfg(test)]
mod tests {
    use crate::check;

    use super::*;

    #[test]
    fn num_signed_unsigned_roundtrip() {
        check(|src| {
            let i: i8 = src.any("i");
            let u = i.unsigned_abs();
            let v = if i.is_negative() {
                <i8 as Int>::from_unsigned(u.wrapping_neg())
            } else {
                <i8 as Int>::from_unsigned(u)
            };
            assert_eq!(i, v);
        });
    }
}
