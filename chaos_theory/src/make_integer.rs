// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    any::TypeId,
    fmt::Debug,
    marker::PhantomData,
    num::{NonZero, Saturating, Wrapping},
    ops::RangeBounds,
    sync::atomic::{
        AtomicBool, AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicIsize, AtomicU8, AtomicU16,
        AtomicU32, AtomicU64, AtomicUsize, Ordering,
    },
};

use crate::{
    Arbitrary, Generator, Int, MaybeOwned, SourceRaw, Tweak, Unsigned as _, math::percent,
    range::Range,
};

const INTEGER_BOUND_PROB: f64 = percent(5);

pub(crate) const BYTE_SPECIAL_PROB: f64 = percent(75); // high percentage because BYTE_SPECIAL is 40% of all possible bytes
pub(crate) const BYTE_SPECIAL: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*-_=+.,;:? \t\r\n/\\|()[]{}<>'\"`\x00\x0B\x1B\x7F";

impl Arbitrary for bool {
    fn arbitrary() -> impl Generator<Item = Self> {
        int_in_range::<u8>(0..=1)
            .map_reversible(|i| i != 0, |b| Some(MaybeOwned::Owned(u8::from(*b))))
    }
}

impl Arbitrary for AtomicBool {
    fn arbitrary() -> impl Generator<Item = Self> {
        bool::arbitrary().map_reversible(Into::into, |b: &Self| {
            Some(MaybeOwned::Owned(b.load(Ordering::Relaxed)))
        })
    }
}

struct Integer<I: Int> {
    range: Range<I>,
    neg: Option<Range<u64>>,
    pos: Option<Range<u64>>,
}

impl<I: Int> Debug for Integer<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple(core::any::type_name::<I>())
            .field(&self.range)
            .finish()
    }
}

impl<I: Int> Generator for Integer<I> {
    type Item = I;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let mut example = example.copied();
        if example.is_none() {
            // Without custom seeds for bytes, we focus too heavy on the ASCII-unprintable range (0-31).
            // We special-case I::MAX below because it will make us generate all-ones pattern.
            let is_bool = self.neg.is_none() && self.pos == Some(Range { min: 0, max: 1 });
            let is_byte = TypeId::of::<I>() == TypeId::of::<u8>() && !is_bool;
            if is_byte {
                let base = BYTE_SPECIAL.len();
                let additional = 2 + usize::from(self.range.max != I::MAX); // don't check if min/max are present
                example = src
                    .as_mut()
                    .choose_seed_index(BYTE_SPECIAL_PROB, base + additional)
                    .map(|seed_ix| match seed_ix.wrapping_sub(base) {
                        0 => self.range.min,
                        1 => self.range.max,
                        2 => I::MAX,
                        _ => {
                            let b = BYTE_SPECIAL[seed_ix];
                            let u = I::Unsigned::from_bits(u64::from(b));
                            I::from_unsigned(u)
                        }
                    });
            } else {
                let seeds: &[I] = if self.range.max == I::MAX {
                    &[self.range.min, self.range.max]
                } else {
                    &[self.range.min, self.range.max, I::MAX]
                };
                example = src.as_mut().choose_seed(INTEGER_BOUND_PROB, seeds).copied();
            }
        }
        let example_neg = example.map(Int::is_negative);
        let (example_neg, forced) = match (self.neg, self.pos) {
            (Some(_), Some(_)) => (example_neg, false),
            (Some(_), None) => (Some(true), true),
            (None, Some(_)) => (Some(false), true),
            (None, None) => unreachable!(),
        };
        if forced {
            src.mark_next_choice_forced();
        }
        let ix = src
            .as_mut()
            .choose_index(2, example_neg.map(usize::from), Tweak::IntSign);
        let r = if ix == 0 { &self.pos } else { &self.neg };
        let u = I::Unsigned::from_bits(src.as_mut().choose_value(
            r.expect("internal error: range not set"),
            example.map(|i| i.unsigned_abs().to_bits()),
            true,
            Tweak::IntValue,
        ));
        if ix == 0 {
            I::from_unsigned(u)
        } else {
            I::from_unsigned(u.wrapping_neg())
        }
    }
}

trait Int128: Copy + Debug {
    type Half: Arbitrary + Copy;

    const MIN: Self;
    const MAX: Self;

    fn lo(self) -> Self::Half;
    fn hi(self) -> Option<Self::Half>;
    fn from_halves(lo: Self::Half, hi: Option<Self::Half>) -> Self;
}

impl Int128 for u128 {
    type Half = u64;

    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;

    fn lo(self) -> Self::Half {
        self as u64
    }

    fn hi(self) -> Option<Self::Half> {
        (self > Self::from(u64::MAX)).then_some((self >> u64::BITS) as u64)
    }

    fn from_halves(lo: Self::Half, hi: Option<Self::Half>) -> Self {
        hi.map_or_else(
            || Self::from(lo),
            |hi| (Self::from(hi) << u64::BITS) | Self::from(lo),
        )
    }
}

impl Int128 for i128 {
    type Half = i64;

    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;

    fn lo(self) -> Self::Half {
        self as i64
    }

    fn hi(self) -> Option<Self::Half> {
        i64::try_from(self)
            .is_err()
            .then_some((self >> i64::BITS) as i64)
    }

    fn from_halves(lo: Self::Half, hi: Option<Self::Half>) -> Self {
        hi.map_or_else(
            || Self::from(lo),
            |hi| (Self::from(hi) << i64::BITS) | Self::from(lo as u64),
        )
    }
}

struct Integer128<I>(PhantomData<I>);

impl<I: Int128> Debug for Integer128<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(core::any::type_name::<I>())
    }
}

impl<I: Int128> Generator for Integer128<I> {
    type Item = I;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let mut example = example.copied();
        if example.is_none() {
            example = src
                .as_mut()
                .choose_seed(INTEGER_BOUND_PROB, &[I::MIN, I::MAX])
                .copied();
        }

        let example_lo = example.map(Int128::lo);
        let lo = src.any("<lo>", example_lo.as_ref());
        let example_hi = example.map(Int128::hi);
        let hi = src.any("<hi>", example_hi.as_ref());
        I::from_halves(lo, hi)
    }
}

impl Arbitrary for u128 {
    fn arbitrary() -> impl Generator<Item = Self> {
        Integer128(PhantomData)
    }
}

impl Arbitrary for i128 {
    fn arbitrary() -> impl Generator<Item = Self> {
        Integer128(PhantomData)
    }
}

macro_rules! impl_arbitrary_int_wrappers {
    ($num: ty) => {
        impl Arbitrary for NonZero<$num> {
            fn arbitrary() -> impl Generator<Item = Self> {
                fn to_nonzero(i: $num) -> NonZero<$num> {
                    NonZero::new(i).expect("internal error: generated zero after filter")
                }

                fn from_nonzero(n: &NonZero<$num>) -> Option<MaybeOwned<'_, $num>> {
                    Some(MaybeOwned::Owned(n.get()))
                }

                <$num>::arbitrary()
                    .filter_assume(|i| *i != 0)
                    .map_reversible(to_nonzero, from_nonzero)
            }
        }

        impl Arbitrary for Saturating<$num> {
            fn arbitrary() -> impl Generator<Item = Self> {
                <$num>::arbitrary().map_reversible(|i| Self(i), |i| Some(MaybeOwned::Owned(i.0)))
            }
        }

        impl Arbitrary for Wrapping<$num> {
            fn arbitrary() -> impl Generator<Item = Self> {
                <$num>::arbitrary().map_reversible(|i| Self(i), |i| Some(MaybeOwned::Owned(i.0)))
            }
        }
    };
}

macro_rules! impl_arbitrary_int {
    ($num: ty, $num_atomic: ty) => {
        impl Arbitrary for $num {
            fn arbitrary() -> impl Generator<Item = Self> {
                int_in_range::<Self>(..)
            }
        }

        impl Arbitrary for $num_atomic {
            fn arbitrary() -> impl Generator<Item = Self> {
                <$num>::arbitrary().map_reversible(Into::into, |i: &Self| {
                    Some(MaybeOwned::Owned(i.load(Ordering::Relaxed)))
                })
            }
        }

        impl_arbitrary_int_wrappers!($num);
    };
}

impl_arbitrary_int!(usize, AtomicUsize);
impl_arbitrary_int!(u8, AtomicU8);
impl_arbitrary_int!(u16, AtomicU16);
impl_arbitrary_int!(u32, AtomicU32);
impl_arbitrary_int!(u64, AtomicU64);

impl_arbitrary_int!(isize, AtomicIsize);
impl_arbitrary_int!(i8, AtomicI8);
impl_arbitrary_int!(i16, AtomicI16);
impl_arbitrary_int!(i32, AtomicI32);
impl_arbitrary_int!(i64, AtomicI64);

impl_arbitrary_int_wrappers!(u128);
impl_arbitrary_int_wrappers!(i128);

/// Create a generator of integers in range.
///
/// If you are generating a size or an index value, prefer
/// [`make::size`](crate::make::size) or [`make::index`](crate::make::index) instead.
pub fn int_in_range<I: Int>(r: impl RangeBounds<I>) -> impl Generator<Item = I> {
    let range = Range::<I>::new(r);
    let (neg_range, pos_range) = range.zero_split();
    Integer {
        range,
        neg: neg_range.map(|r| {
            Range::new_raw(
                r.max.unsigned_abs().to_bits(),
                r.min.unsigned_abs().to_bits(),
            )
        }),
        pos: pos_range.map(|r| {
            Range::new_raw(
                r.min.unsigned_abs().to_bits(),
                r.max.unsigned_abs().to_bits(),
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Env, Source, check,
        tests::{print_debug_examples, prop_smoke},
    };

    #[test]
    fn integer_smoke() {
        check(|src| {
            prop_smoke(src, "bool", bool::arbitrary());

            prop_smoke(src, "u8", u8::arbitrary());
            prop_smoke(src, "u16", u16::arbitrary());
            prop_smoke(src, "u32", u32::arbitrary());
            prop_smoke(src, "u64", u64::arbitrary());
            prop_smoke(src, "u128", u128::arbitrary());
            prop_smoke(src, "usize", usize::arbitrary());

            prop_smoke(src, "i8", i8::arbitrary());
            prop_smoke(src, "i16", i16::arbitrary());
            prop_smoke(src, "i32", i32::arbitrary());
            prop_smoke(src, "i64", i64::arbitrary());
            prop_smoke(src, "i128", i128::arbitrary());
            prop_smoke(src, "isize", isize::arbitrary());
        });
    }

    #[test]
    fn integer_gen_in_range() {
        fn prop_gen_in_range<I: Int + Arbitrary>(src: &mut Source, label: &'static str) {
            src.scope(label, |src| {
                let r: Range<I> = src.any("r");
                let g = int_in_range::<I>(r);
                let value = src.any_of("value", &g);
                assert!(r.contains(&value));
                prop_smoke(src, label, &g);
            });
        }

        check(|src| {
            prop_gen_in_range::<u8>(src, "u8");
            prop_gen_in_range::<u16>(src, "u16");
            prop_gen_in_range::<u32>(src, "u32");
            prop_gen_in_range::<u64>(src, "u64");
            prop_gen_in_range::<usize>(src, "usize");

            prop_gen_in_range::<i8>(src, "i8");
            prop_gen_in_range::<i16>(src, "i16");
            prop_gen_in_range::<i32>(src, "i32");
            prop_gen_in_range::<i64>(src, "i64");
            prop_gen_in_range::<isize>(src, "isize");
        });
    }

    #[test]
    fn integer_bound_coverage() {
        fn prop_bound_coverage<I: Int + Arbitrary>(src: &mut Source, label: &str) {
            src.scope(label, |src| {
                let r: Range<I> = src.any("r");
                let g = int_in_range::<I>(r);
                let base_seed: u32 = src.any("base_seed");
                let (mut got_min, mut got_max, mut got_zero) =
                    (false, false, !r.contains(&I::ZERO));
                for s in 0..64 {
                    let seed = base_seed.wrapping_add(s);
                    let mut env = Env::custom().with_rng_seed(seed).env(false);
                    let mut src = Source::new(&mut env);
                    for _ in 0..64 {
                        let i = g.next(src.as_raw(), None);
                        got_min = got_min || i == r.min;
                        got_max = got_max || i == r.max;
                        got_zero = got_zero || i == I::ZERO;
                        if got_min && got_max && got_zero {
                            return;
                        }
                    }
                }
                assert!(got_min);
                assert!(got_max);
                assert!(got_zero);
            });
        }

        check(|src| {
            prop_bound_coverage::<u8>(src, "u8");
            prop_bound_coverage::<u16>(src, "u16");
            prop_bound_coverage::<u32>(src, "u32");
            prop_bound_coverage::<u64>(src, "u64");
            prop_bound_coverage::<usize>(src, "usize");

            prop_bound_coverage::<i8>(src, "i8");
            prop_bound_coverage::<i16>(src, "i16");
            prop_bound_coverage::<i32>(src, "i32");
            prop_bound_coverage::<i64>(src, "i64");
            prop_bound_coverage::<isize>(src, "isize");
        });
    }

    #[test]
    fn bool_examples() {
        print_debug_examples(bool::arbitrary(), None, Ord::cmp);
    }

    #[test]
    fn byte_examples() {
        print_debug_examples(u8::arbitrary().map(|b| (b, char::from(b))), None, Ord::cmp);
    }

    #[test]
    fn integer_examples() {
        let gens = [
            int_in_range::<i64>(-3..).boxed(),
            int_in_range::<i64>(..=3).boxed(),
            int_in_range::<i64>(-1000..=1_000_000).boxed(),
            int_in_range::<i64>(-3..=7).boxed(),
            i64::arbitrary().boxed(),
        ];
        for g in gens {
            print_debug_examples(g, None, Ord::cmp);
        }
    }

    #[test]
    fn i128_examples() {
        print_debug_examples(i128::arbitrary(), None, Ord::cmp);
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use core::hint::black_box;

    use crate::{Arbitrary as _, make, tests::bench_gen_next};

    #[bench]
    fn gen_bool(b: &mut test::Bencher) {
        let g = bool::arbitrary();
        bench_gen_next(b, &g);
    }

    #[bench]
    fn gen_u64_arbitrary(b: &mut test::Bencher) {
        let g = u64::arbitrary();
        bench_gen_next(b, &g);
    }

    #[bench]
    fn gen_u64_unbounded(b: &mut test::Bencher) {
        let g = make::int_in_range::<u64>(black_box(..));
        bench_gen_next(b, &g);
    }

    #[bench]
    fn gen_i64_arbitrary(b: &mut test::Bencher) {
        let g = i64::arbitrary();
        bench_gen_next(b, &g);
    }
}
