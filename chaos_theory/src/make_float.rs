// Copyright 2025 Ilya Shcherbak <tthread@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{fmt::Debug, ops::RangeBounds};

use crate::{
    Arbitrary, Float, Generator, SourceRaw, Tweak, Unsigned as _,
    make::int_in_range,
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
    fn new<F: Float>(r: Range<F>, example: Option<F>) -> Self {
        debug_assert!(r.min >= F::ZERO && r.max >= F::ZERO);
        let (exp_min, sig_int_min, sig_frac_min) = extract_float_parts(r.min);
        let (exp_max, sig_int_max, sig_frac_max) = extract_float_parts(r.max);
        let example = example.map(|f| extract_float_parts(f));
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
    range: Range<F>,
    neg: Option<Range<F>>,
    pos: Option<Range<F>>,
}

impl<F: Float> Generator for Floating<F> {
    type Item = F;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let mut example = example.copied();
        if example.is_none() {
            // TODO: generate other interesting values as well
            example = src
                .as_mut()
                .choose_seed(FLOAT_BOUND_PROB, &[F::ZERO, self.range.min, self.range.max])
                .copied();
        }
        let example_neg = example.map(Float::is_negative);
        let (example_neg, forced) = match (self.neg, self.pos) {
            (Some(_), Some(_)) => (example_neg, false),
            (None, Some(_)) => (Some(false), true),
            (Some(_), None) => (Some(true), true),
            _ => unreachable!("internal error: impossible range combination"),
        };
        if forced {
            src.mark_next_choice_forced();
        }
        let ix = src
            .as_mut()
            .choose_index(2, example_neg.map(usize::from), Tweak::FloatSign);
        let range = if ix == 0 { &self.pos } else { &self.neg };
        let f = F::from_bits(gen_unsigned_float(
            src,
            range.expect("internal error: range not set"),
            example,
        ));
        if ix == 0 { f } else { f.negate() }
    }
}

// TODO: NaN generation
//
// Plan:
// - create OrderedFloat wrapper that considers NaN as signed and more than Inf
// - customize Ranged (next_up/next_down) for OrderedFloat
// - use OrderedFloat in this file
// - NaN becomes a non-special case (both Inf and NaN appear in seeds and/or bounds)

impl Arbitrary for f32 {
    fn arbitrary() -> impl Generator<Item = Self> {
        float_in_range(..)
    }
}

impl Arbitrary for f64 {
    fn arbitrary() -> impl Generator<Item = Self> {
        float_in_range(..)
    }
}

/// Create a generator of floats in range.
pub fn float_in_range<F: Float>(r: impl RangeBounds<F>) -> impl Generator<Item = F> {
    let range = Range::new(r);
    let (neg, pos) = range.zero_split();
    Floating {
        range,
        neg: neg.map(|r| Range::new_raw(r.max.negate(), r.min.negate())),
        pos,
    }
}

fn gen_unsigned_float<F: Float>(src: &mut SourceRaw, r: Range<F>, example: Option<F>) -> u64 {
    let r = &FloatRange::new(r, example);
    let e = choose_exp(src, r);
    let si = choose_sig_int::<F>(src, e, r);
    let sf = choose_sig_frac::<F>(src, e, si, r);
    compose_float::<F>(e, si, sf)
}

fn extract_float_parts<F: Float>(f: F) -> (i32, u64, u64) {
    let u = f.to_bits_unsigned();
    let exp = (u >> F::MANTISSA_BITS) as i32 - F::EXPONENT_BIAS;
    let frac = u & bitmask_u64(F::MANTISSA_BITS);
    let n = frac_bits::<F>(exp);
    (exp, frac >> n, u & bitmask_u64(n))
}

fn compose_float<F: Float>(exp: i32, sig_int: u64, sig_frac: u64) -> u64 {
    let e = exp + F::EXPONENT_BIAS;
    let e = (e as u64) << F::MANTISSA_BITS;
    let s = (sig_int << frac_bits::<F>(exp)) | sig_frac;
    e | s
}

fn choose_exp(src: &mut SourceRaw, r: &FloatRange) -> i32 {
    int_in_range(r.exp_min..=r.exp_max).next(src, r.example.map(|e| e.0).as_ref())
}

fn choose_sig_int<F: Float>(src: &mut SourceRaw, exp: i32, r: &FloatRange) -> u64 {
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

fn choose_sig_frac<F: Float>(src: &mut SourceRaw, exp: i32, sig_int: u64, r: &FloatRange) -> u64 {
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
    let mut sig_frac = src
        .as_mut()
        .choose_value(range, r.example.map(|e| e.2), false);
    let range = Range::new_raw(min.bit_len() as u64, sig_frac.bit_len() as u64);
    let sig_frac_bits =
        src.as_mut()
            .choose_value(range, r.example.map(|_| sig_frac.bit_len() as u64), false);
    let bits_to_mask = sig_frac.bit_len() as u64 - sig_frac_bits;
    for i in 0..bits_to_mask {
        let sig_frac_masked = sig_frac & (!(1u64 << i));
        if sig_frac_masked < min {
            break;
        }
        sig_frac = sig_frac_masked;
    }
    sig_frac
}

fn frac_bits<F: Float>(exp: i32) -> u64 {
    let m = F::MANTISSA_BITS as i32;
    (m - exp).clamp(0, m) as u64
}

fn bitmask_u64(u: u64) -> u64 {
    math::bitmask::<u64>(u as usize)
}

#[cfg(test)]
#[expect(clippy::float_cmp)]
mod tests {
    use crate::{
        Env, Float, Generator as _, Source, check,
        make_float::{Arbitrary, compose_float, extract_float_parts, float_in_range},
        range::Range,
        slow_test_enabled,
        tests::{print_debug_examples, prop_smoke},
    };
    use core::{cmp::Ordering, ops::RangeBounds as _};

    #[test]
    #[expect(clippy::similar_names)]
    fn f32_full_scan_extract_compose() {
        if !slow_test_enabled() {
            return;
        }
        for i in 0..i32::MAX {
            let f = f32::from_bits(i as u32);
            let f_neg = f32::from_bits(i as u32).negate();
            let (e, si, sf) = extract_float_parts(f);
            let (e_neg, si_neg, sf_neg) = extract_float_parts(f_neg);
            let a_f = f32::from_bits(compose_float::<f32>(e, si, sf) as u32);
            let a_f_neg =
                f32::from_bits(compose_float::<f32>(e_neg, si_neg, sf_neg) as u32).negate();
            if f == f {
                assert_eq!(f, a_f);
                assert_eq!(f_neg, a_f_neg);
            }
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
                let mut a_f = F::from_bits(compose_float::<F>(e, si, sf));
                if f.is_negative() {
                    a_f = a_f.negate();
                }
                assert_eq!(f, a_f);
            });
        }
        check(|src| {
            extract_compose_impl_test::<f32>(src, "f32");
            extract_compose_impl_test::<f64>(src, "f64");
        });
    }

    #[test]
    fn float_smoke() {
        check(|src| {
            prop_smoke(src, "f32", f32::arbitrary());
            prop_smoke(src, "f64", f64::arbitrary());
        });
    }

    #[test]
    fn float_gen_in_range() {
        fn gen_in_range<F: Float + Arbitrary>(src: &mut Source<'_>, label: &'static str) {
            src.scope(label, |src| {
                let r: Range<F> = src.any("r");
                let g = float_in_range(r);
                let value = src.any_of("value", &g);
                assert!(r.contains(&value));
                prop_smoke(src, label, &g);
            });
        }
        check(|src| {
            gen_in_range::<f32>(src, "f32");
            gen_in_range::<f64>(src, "f64");
        });
    }

    #[test]
    fn range_coverage_test() {
        fn range_coverage_test_impl<F: Float + Arbitrary>(src: &mut Source) {
            let r: Range<F> = src.any("r");
            let g = float_in_range(r);
            let base_seed: u32 = src.any("base_seed");
            let (mut got_max, mut got_min, mut got_zero) = (false, false, !r.contains(&F::ZERO));
            for s in 0..512 {
                let seed = base_seed.wrapping_add(s);
                let mut env = Env::custom().with_rng_seed(seed).env(false);
                let mut src = Source::new(&mut env);
                let f = g.next(src.as_raw(), None);
                got_max = got_max || f == r.max;
                got_min = got_min || f == r.min;
                got_zero = got_zero || f == F::ZERO;
            }
            assert!(got_min);
            assert!(got_max);
            assert!(got_zero);
        }

        check(|src| {
            src.scope("f32", |src| {
                range_coverage_test_impl::<f32>(src);
            });
            src.scope("f64", |src| {
                range_coverage_test_impl::<f64>(src);
            });
        });
    }

    fn float_cmp<F: Float>(a: &F, b: &F) -> Ordering {
        if a == b {
            Ordering::Equal
        } else if a < b {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }

    #[test]
    fn f32_examples() {
        let gens = [
            float_in_range::<f32>(-3.0..).boxed(),
            float_in_range::<f32>(..=3.5).boxed(),
            float_in_range::<f32>(1000.0..=1_000_000.0).boxed(),
            float_in_range::<f32>(-4.5..=9.9).boxed(),
            float_in_range::<f32>(0.0..1.0).boxed(),
            f32::arbitrary().boxed(),
        ];
        for g in gens {
            print_debug_examples(g, None, float_cmp);
        }
    }

    #[test]
    fn f64_examples() {
        let gens = [
            float_in_range::<f64>(-3.0..).boxed(),
            float_in_range::<f64>(..=3.5).boxed(),
            float_in_range::<f64>(1000.0..=1_000_000.0).boxed(),
            float_in_range::<f64>(-4.5..=9.9).boxed(),
            float_in_range::<f64>(0.0..1.0).boxed(),
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
