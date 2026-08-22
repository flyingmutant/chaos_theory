// Copyright 2025 Ilya Shcherbak <tthread@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{fmt::Debug, ops::RangeBounds};

use crate::{
    Arbitrary, Float, Generator, SourceEx, Unsigned as _,
    env::Tweak,
    make::int_in,
    math::{self, percent},
    range::Range,
};

// TODO: because we operate on biased exponent, we reduce floats to 1.0 and not 0.0

const FLOAT_BOUND_PROB: f64 = percent(5);

struct FloatRange {
    exp_min: i32,
    exp_max: i32,
    sig_int_min: u64,
    sig_int_max: u64,
    sig_frac_min: u64,
    sig_frac_max: u64,
    example: Option<(i32, u64, u64)>,
}

impl FloatRange {
    fn new<F: Float>(r: Range<u64>, example: Option<u64>) -> Self {
        let (exp_min, sig_int_min, sig_frac_min) = extract_magnitude_parts::<F>(r.min);
        let (exp_max, sig_int_max, sig_frac_max) = extract_magnitude_parts::<F>(r.max);
        let example = example.map(extract_magnitude_parts::<F>);
        Self {
            exp_min,
            exp_max,
            sig_int_min,
            sig_int_max,
            sig_frac_min,
            sig_frac_max,
            example,
        }
    }
}

#[derive(Debug)]
struct Floating<F: Float> {
    // Magnitudes are ordered exactly like non-negative floats, continuing from infinity through
    // every NaN payload. The sign ranges decide which signs are valid for each magnitude.
    magnitudes: Range<u64>,
    neg: Option<Range<u64>>,
    pos: Option<Range<u64>>,
    seed_min: F,
    seed_max: F,
}

impl<F: Float> Floating<F> {
    fn new(neg: Option<Range<u64>>, pos: Option<Range<u64>>) -> Self {
        let infinity = F::MAX.to_bits_unsigned();
        let magnitudes = match (neg, pos) {
            (Some(neg), Some(pos)) => {
                // Both signs are only present for ranges that cross zero (or for the complete
                // bit-pattern domain), so their magnitude ranges overlap and have no gap.
                debug_assert!(neg.min <= pos.max && pos.min <= neg.max);
                Range::new_raw(neg.min.min(pos.min), neg.max.max(pos.max))
            }
            (Some(r), None) | (None, Some(r)) => r,
            (None, None) => unreachable!("internal error: impossible range combination"),
        };
        let seed_min = match (neg, pos) {
            (Some(r), _) => from_magnitude(r.max.min(infinity), true),
            (None, Some(r)) => from_magnitude(r.min, false),
            (None, None) => unreachable!("internal error: impossible range combination"),
        };
        let seed_max = match (neg, pos) {
            (_, Some(r)) => from_magnitude(r.max.min(infinity), false),
            (Some(r), None) => from_magnitude(r.min, true),
            (None, None) => unreachable!("internal error: impossible range combination"),
        };
        Self {
            magnitudes,
            neg,
            pos,
            seed_min,
            seed_max,
        }
    }
}

impl<F: Float> Generator for Floating<F> {
    type Item = F;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let mut example = example.copied();
        if example.is_none() {
            // TODO: generate other interesting values as well
            example = src
                .as_mut()
                .choose_seed(
                    FLOAT_BOUND_PROB,
                    &[F::ZERO, F::ZERO.negate(), self.seed_min, self.seed_max],
                )
                .copied();
        }
        // Choose magnitude before sign so a zero-only sign range does not make zero dominate a
        // one-sided numeric range. The same choices are made for arbitrary and ranged floats.
        let magnitude =
            gen_unsigned_float::<F>(src, self.magnitudes, example.map(Float::to_bits_unsigned));
        let neg_allowed = self
            .neg
            .is_some_and(|r| magnitude >= r.min && magnitude <= r.max);
        let pos_allowed = self
            .pos
            .is_some_and(|r| magnitude >= r.min && magnitude <= r.max);
        let example_neg = example.map(Float::is_sign_negative);
        let (example_neg, forced) = match (neg_allowed, pos_allowed) {
            (true, true) => (example_neg, false),
            (false, true) => (Some(false), true),
            (true, false) => (Some(true), true),
            (false, false) => unreachable!("internal error: magnitude outside sign ranges"),
        };
        if forced {
            src.mark_next_choice_forced();
        }
        let ix = src
            .as_mut()
            .choose_index(2, example_neg.map(usize::from), Tweak::FloatSign);
        from_magnitude(magnitude, ix != 0)
    }
}

impl Arbitrary for f32 {
    fn arbitrary() -> impl Generator<Item = Self> {
        arbitrary_float()
    }
}

impl Arbitrary for f64 {
    fn arbitrary() -> impl Generator<Item = Self> {
        arbitrary_float()
    }
}

fn arbitrary_float<F: Float>() -> Floating<F> {
    // Unlike numeric ranges, Arbitrary covers every IEEE bit pattern, including signed NaNs.
    let magnitudes = Range::new_raw(0, magnitude_mask::<F>());
    Floating::new(Some(magnitudes), Some(magnitudes))
}

/// Create a generator of non-NaN floats in a numeric range.
///
/// If the range contains zero, both positive and negative zero can be generated.
///
/// # Panics
///
/// Panics if the range is empty or either bound is NaN.
pub fn float_in<F: Float>(range: impl RangeBounds<F>) -> impl Generator<Item = F> {
    let range = Range::new(range);
    let contains_zero = range.min <= F::ZERO && range.max >= F::ZERO;
    let neg = if range.min < F::ZERO {
        let min = if contains_zero {
            0
        } else {
            range.max.to_bits_unsigned()
        };
        Some(Range::new_raw(min, range.min.to_bits_unsigned()))
    } else if contains_zero {
        Some(Range::new_raw(0, 0))
    } else {
        None
    };
    let pos = if range.max > F::ZERO {
        let min = if contains_zero {
            0
        } else {
            range.min.to_bits_unsigned()
        };
        Some(Range::new_raw(min, range.max.to_bits_unsigned()))
    } else if contains_zero {
        Some(Range::new_raw(0, 0))
    } else {
        None
    };
    Floating::new(neg, pos)
}

fn gen_unsigned_float<F: Float>(src: &mut SourceEx, r: Range<u64>, example: Option<u64>) -> u64 {
    let r = &FloatRange::new::<F>(r, example);
    let e = choose_exp(src, r);
    let si = choose_sig_int::<F>(src, e, r);
    let sf = choose_sig_frac::<F>(src, e, si, r);
    compose_float::<F>(e, si, sf)
}

#[cfg(test)]
fn extract_float_parts<F: Float>(f: F) -> (i32, u64, u64) {
    extract_magnitude_parts::<F>(f.to_bits_unsigned())
}

fn extract_magnitude_parts<F: Float>(u: u64) -> (i32, u64, u64) {
    let exp = (u >> F::MANTISSA_BITS) as i32 - F::EXPONENT_BIAS;
    let frac = u & bitmask_u64(F::MANTISSA_BITS);
    let n = frac_bits::<F>(exp);
    (exp, frac >> n, u & bitmask_u64(n))
}

fn from_magnitude<F: Float>(magnitude: u64, negative: bool) -> F {
    debug_assert_eq!(magnitude & !magnitude_mask::<F>(), 0);
    let sign = u64::from(negative) << (size_of::<F>() * 8 - 1);
    F::from_bits(sign | magnitude)
}

fn magnitude_mask<F: Float>() -> u64 {
    bitmask_u64((size_of::<F>() * 8 - 1) as u64)
}

fn compose_float<F: Float>(exp: i32, sig_int: u64, sig_frac: u64) -> u64 {
    let e = exp + F::EXPONENT_BIAS;
    let e = (e as u64) << F::MANTISSA_BITS;
    let s = (sig_int << frac_bits::<F>(exp)) | sig_frac;
    e | s
}

fn choose_exp(src: &mut SourceEx, r: &FloatRange) -> i32 {
    int_in(r.exp_min..=r.exp_max).next(src, r.example.map(|e| e.0).as_ref())
}

fn choose_sig_int<F: Float>(src: &mut SourceEx, exp: i32, r: &FloatRange) -> u64 {
    let (min, max) = if r.exp_min == r.exp_max {
        (r.sig_int_min, r.sig_int_max)
    } else if exp == r.exp_min {
        (
            r.sig_int_min,
            bitmask_u64(F::MANTISSA_BITS - frac_bits::<F>(exp)),
        )
    } else if exp == r.exp_max {
        (0, r.sig_int_max)
    } else {
        (0, bitmask_u64(F::MANTISSA_BITS - frac_bits::<F>(exp)))
    };
    let range = Range::new_raw(min, max);
    src.as_mut()
        .choose_value(range, r.example.map(|e| e.1), false)
}

fn choose_sig_frac<F: Float>(src: &mut SourceEx, exp: i32, sig_int: u64, r: &FloatRange) -> u64 {
    let (min, max) = if r.exp_min == r.exp_max && r.sig_int_min == r.sig_int_max {
        (r.sig_frac_min, r.sig_frac_max)
    } else if exp == r.exp_min && sig_int == r.sig_int_min {
        (r.sig_frac_min, bitmask_u64(frac_bits::<F>(exp)))
    } else if exp == r.exp_max && sig_int == r.sig_int_max {
        (0, r.sig_frac_max)
    } else {
        (0, bitmask_u64(frac_bits::<F>(exp)))
    };
    let range = Range::new_raw(min, max);
    let sig_frac = src
        .as_mut()
        .choose_value(range, r.example.map(|e| e.2), false);
    let total_sig_frac_bits = sig_frac.bit_len() as u64;
    let range = Range::new_raw(
        total_sig_frac_bits - max_bits_to_mask(sig_frac, min),
        total_sig_frac_bits,
    );
    let sig_frac_bits =
        src.as_mut()
            .choose_value(range, r.example.map(|_| total_sig_frac_bits), false);
    sig_frac & !bitmask_u64(total_sig_frac_bits - sig_frac_bits)
}

fn max_bits_to_mask(sig_frac: u64, min: u64) -> u64 {
    let mut bits = 0;
    while bits < sig_frac.bit_len() as u64 {
        let sig_frac_masked = sig_frac & !bitmask_u64(bits + 1);
        if sig_frac_masked < min {
            break;
        }
        bits += 1;
    }
    bits
}

fn frac_bits<F: Float>(exp: i32) -> u64 {
    let m = F::MANTISSA_BITS as i32;
    (m - exp).clamp(0, m) as u64
}

fn bitmask_u64(u: u64) -> u64 {
    math::bitmask::<u64>(u as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Env, Float, Source, check,
        config::slow_test_enabled,
        make_float::{Arbitrary, compose_float, extract_float_parts, float_in},
        range::Range,
        tests::{print_debug_examples, prop_smoke_by},
    };
    use core::cmp::Ordering;

    fn same_float<F: Float>(a: &F, b: &F) -> bool {
        a.to_bits() == b.to_bits()
    }

    #[test]
    #[expect(clippy::similar_names)]
    fn f32_full_scan_extract_compose() {
        if !slow_test_enabled() {
            return;
        }
        for i in 0..i32::MAX {
            let f = f32::from_bits(i as u32);
            let f_neg = -f;
            let (e, si, sf) = extract_float_parts(f);
            let (e_neg, si_neg, sf_neg) = extract_float_parts(f_neg);
            let a_f = f32::from_bits(compose_float::<f32>(e, si, sf) as u32);
            let a_f_neg = -f32::from_bits(compose_float::<f32>(e_neg, si_neg, sf_neg) as u32);
            assert_eq!(f.to_bits(), a_f.to_bits());
            assert_eq!(f_neg.to_bits(), a_f_neg.to_bits());
        }
    }

    #[test]
    fn extract_compose() {
        fn extract_compose_impl_test<F: Float + Arbitrary>(
            src: &mut Source<'_>,
            label: &'static str,
        ) {
            src.scope(label, |src| {
                let f: F = src.any("f");
                let (e, si, sf) = extract_float_parts(f);
                let a_f = from_magnitude::<F>(compose_float::<F>(e, si, sf), f.is_sign_negative());
                assert_eq!(f.to_bits(), a_f.to_bits());
            });
        }
        check(|src| {
            extract_compose_impl_test::<f32>(src, "f32");
            extract_compose_impl_test::<f64>(src, "f64");
        });
    }

    #[test]
    fn sig_frac_masking_respects_exact_lower_bound() {
        let sig_frac = 0b111_1011u64;
        let min = 0b110_0100u64;
        let sig_frac_bits = sig_frac.bit_len() as u64;
        let min_sig_frac_bits = sig_frac_bits - max_bits_to_mask(sig_frac, min);

        assert_eq!(min_sig_frac_bits, 3);
        assert_eq!(
            sig_frac & !bitmask_u64(sig_frac_bits - min_sig_frac_bits),
            0b111_0000
        );
        assert_eq!(max_bits_to_mask(0b110_0100, 0b110_0100), 2);
        assert_eq!(max_bits_to_mask(0b111_1011, 0), 7);
    }

    #[test]
    fn float_smoke() {
        check(|src| {
            prop_smoke_by(src, "f32", f32::arbitrary(), same_float);
            prop_smoke_by(src, "f64", f64::arbitrary(), same_float);
        });
    }

    fn next_like<F: Float>(g: &impl Generator<Item = F>, example: F) -> F {
        let mut env = Env::custom().with_rng_budget(usize::MAX).env(false);
        let mut src = Source::new(&mut env);
        g.next(src.as_ex(), Some(&example))
    }

    #[test]
    fn arbitrary_float_reconstructs_all_bit_pattern_classes() {
        fn reconstruct<F: Float>() {
            let g = arbitrary_float::<F>();
            let sign = 1 << (size_of::<F>() * 8 - 1);
            let infinity = F::MAX.to_bits_unsigned();
            let quiet = 1 << (F::MANTISSA_BITS - 1);
            let magnitude_max = magnitude_mask::<F>();
            let patterns = [
                0,
                sign,
                infinity,
                sign | infinity,
                infinity | 1,
                sign | infinity | 1,
                infinity | quiet,
                sign | infinity | quiet,
                magnitude_max,
                sign | magnitude_max,
            ];
            for bits in patterns {
                let example = F::from_bits(bits);
                let value = next_like(&g, example);
                assert_eq!(value.to_bits(), bits);
            }
        }

        reconstruct::<f32>();
        reconstruct::<f64>();
    }

    #[test]
    fn numeric_ranges_treat_zero_signs_as_the_same_endpoint() {
        let zero = float_in::<f32>(0.0..=0.0);
        assert_eq!(next_like(&zero, 0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(next_like(&zero, -0.0).to_bits(), (-0.0f32).to_bits());

        let positive =
            float_in::<f32>((core::ops::Bound::Excluded(0.0), core::ops::Bound::Unbounded));
        assert!(next_like(&positive, -0.0) > 0.0);

        let negative = float_in::<f32>((
            core::ops::Bound::Unbounded,
            core::ops::Bound::Excluded(-0.0),
        ));
        assert!(next_like(&negative, 0.0) < 0.0);
    }

    #[test]
    #[should_panic(expected = "invalid range")]
    fn numeric_range_rejects_nan_bound() {
        let _ = float_in(f32::NAN..);
    }

    #[test]
    fn float_gen_in_range() {
        fn gen_in_range<F: Float + Arbitrary>(src: &mut Source<'_>, label: &'static str) {
            src.scope(label, |src| {
                let r: Range<F> = src.any("r");
                let g = float_in(r);
                let value = src.any_of("value", &g);
                assert!(r.contains(&value));
                prop_smoke_by(src, label, &g, same_float);
            });
        }
        check(|src| {
            gen_in_range::<f32>(src, "f32");
            gen_in_range::<f64>(src, "f64");
        });
    }

    #[test]
    fn range_coverage_test() {
        fn range_coverage_test_impl<F: Float + Arbitrary>(src: &mut Source, label: &str) {
            src.scope(label, |src| {
                let r: Range<F> = src.any("r");
                let g = float_in(r);
                let base_seed: u32 = src.any("base_seed");
                let (mut got_max, mut got_min, mut got_zero) =
                    (false, false, !r.contains(&F::ZERO));
                for s in 0..64 {
                    let seed = base_seed.wrapping_add(s);
                    let mut env = Env::custom().with_rng_seed(seed).env(false);
                    let mut src = Source::new(&mut env);
                    for _ in 0..64 {
                        let f = g.next(src.as_ex(), None);
                        got_max = got_max || f == r.max;
                        got_min = got_min || f == r.min;
                        got_zero = got_zero || f == F::ZERO;
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
            range_coverage_test_impl::<f32>(src, "f32");
            range_coverage_test_impl::<f64>(src, "f64");
        });
    }

    fn float_cmp<F: Float>(a: &F, b: &F) -> Ordering {
        match (a.is_sign_negative(), b.is_sign_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (true, true) => b.to_bits_unsigned().cmp(&a.to_bits_unsigned()),
            (false, false) => a.to_bits_unsigned().cmp(&b.to_bits_unsigned()),
        }
    }

    #[test]
    fn f32_examples() {
        let gens = [
            float_in::<f32>(-3.0..).boxed(),
            float_in::<f32>(..=3.5).boxed(),
            float_in::<f32>(1000.0..=1_000_000.0).boxed(),
            float_in::<f32>(-4.5..=9.9).boxed(),
            float_in::<f32>(0.0..1.0).boxed(),
            f32::arbitrary().boxed(),
        ];
        for g in gens {
            print_debug_examples(g, None, float_cmp);
        }
    }

    #[test]
    fn f64_examples() {
        let gens = [
            float_in::<f64>(-3.0..).boxed(),
            float_in::<f64>(..=3.5).boxed(),
            float_in::<f64>(1000.0..=1_000_000.0).boxed(),
            float_in::<f64>(-4.5..=9.9).boxed(),
            float_in::<f64>(0.0..1.0).boxed(),
            f64::arbitrary().boxed(),
        ];
        for g in gens {
            print_debug_examples(g, None, float_cmp);
        }
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use crate::{Arbitrary as _, tests::bench_gen_next};

    #[bench]
    fn gen_f32_full_range(b: &mut test::Bencher) {
        let g = f32::arbitrary();
        bench_gen_next(b, &g);
    }

    #[bench]
    fn gen_f64_full_range(b: &mut test::Bencher) {
        let g = f64::arbitrary();
        bench_gen_next(b, &g);
    }
}
