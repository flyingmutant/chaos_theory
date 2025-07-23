// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{
    TEMPERATURE_BOUND_EXCLUSIVE, TEMPERATURE_DEFAULT,
    rand::{Rand, RandCore},
};

pub(crate) fn temperature_scale_down(t: u8) -> u8 {
    let t = u32::from(t);
    let t = t
        .saturating_sub(u32::from(TEMPERATURE_DEFAULT))
        .saturating_mul(u32::from(TEMPERATURE_BOUND_EXCLUSIVE))
        .saturating_div(u32::from(TEMPERATURE_BOUND_EXCLUSIVE) - u32::from(TEMPERATURE_DEFAULT));
    u8::try_from(t).unwrap_or(u8::MAX)
}

#[derive(Debug)]
pub(crate) struct Biased {
    v: f64,
    q: f64,
}

impl Biased {
    #[cfg(test)]
    const V_DEFAULT: f64 = 8.0;
    const V_MIN: f64 = 1.0;
    const V_MAX: f64 = 16.0;

    fn v_for_temperature(t: u8) -> f64 {
        let t = f64::from(t) / f64::from(TEMPERATURE_BOUND_EXCLUSIVE);
        debug_assert!((0.0..1.0).contains(&t));
        (Self::V_MAX * t).max(Self::V_MIN)
    }

    fn new(v: f64, q: f64) -> Self {
        debug_assert!(v > 0.0);
        debug_assert!(q > 1.0);
        Self { v, q }
    }

    pub(crate) fn new_temperature(t: u8, q: Option<f64>) -> Self {
        Self::new(Self::v_for_temperature(t), q.unwrap_or(DEFAULT_Q))
    }

    pub(crate) fn sample(&self, rng: &mut Rand<impl RandCore>, bound: usize) -> usize {
        match bound {
            0 => unreachable!(),
            1 => 0,
            2 => {
                // Fast-path with linear approximation for bool/Option.
                let p = self.v.mul_add(-0.065, 0.85).max(0.5);
                usize::from(rng.next_float() > p)
            }
            _ => {
                let v = rng.next_float();
                let n = Zipf::new(bound as u64 - 1, Some(self.v), Some(self.q)).sample_oneshot(v);
                n as usize
            }
        }
    }
}

#[cfg(test)]
struct Geometric {
    log1m_p: f64,
}

#[cfg(test)]
impl Geometric {
    fn new_with_mean(m: f64) -> Self {
        debug_assert!(m >= 0.0);
        let p = 1.0 / (m + 1.0);
        Self::new(p)
    }

    fn new(p: f64) -> Self {
        debug_assert!(p > 0.0 && p <= 1.0);
        Self {
            log1m_p: (-p).ln_1p(),
        }
    }

    fn sample(&self, rng: &mut Rand<impl RandCore>) -> u64 {
        let f = rng.next_float();
        let n = (-f).ln_1p() / self.log1m_p;
        n as u64
    }
}

#[cfg(test)]
fn bound_wrap(v: u64, max: u64) -> u64 {
    if v <= max {
        v
    } else {
        let e = v - max - 1;
        max.saturating_sub(e)
    }
}

// P(x) = (v + x) ^ -q
//
// W.Hormann, G.Derflinger: "Rejection-Inversion to Generate Variates from Monotone Discrete Distributions".
// See:
// - Original paper: http://eeyore.wu-wien.ac.at/papers/96-04-04.wh-der.ps.gz
// - Go code: https://github.com/golang/go/blob/go1.22.3/src/math/rand/v2/zipf.go
// - Abseil code: https://github.com/abseil/abseil-cpp/blob/lts_2024_01_16/absl/random/zipf_distribution.h
struct Zipf {
    k: f64,
    q: f64,
    v: f64,

    one_minus_q: f64,     // 1-q
    one_minus_q_inv: f64, // 1 / 1-q
    hxm: f64,             // h(k + 0.5)
    hx0_minus_hxm: f64,   // h(x0) - h(k + 0.5)
    s: f64,
}

const DEFAULT_Q: f64 = 2.0;
const DEFAULT_V: f64 = 1.0;
const DEFAULT_ONE_MINUS_Q: f64 = 1.0 - DEFAULT_Q;

impl Zipf {
    // Generates values in [0, k].
    fn new(k: u64, v: Option<f64>, q: Option<f64>) -> Self {
        let v = v.unwrap_or(DEFAULT_V);
        let q = q.unwrap_or(DEFAULT_Q);
        debug_assert!(v > 0.0);
        debug_assert!(q > 1.0);

        let k = k as f64;
        let use_precomputed = q == DEFAULT_Q && v == DEFAULT_V;
        let one_minus_q = 1.0 - q;
        let one_minus_q_inv = 1.0 / one_minus_q;

        let mut z = Self {
            k,
            q,
            v,
            one_minus_q,
            one_minus_q_inv,
            hxm: 0.0,
            hx0_minus_hxm: 0.0,
            s: 0.0,
        };

        let h0x5 = if use_precomputed {
            -1.0 / 1.5 // exp(-log(1.5))
        } else {
            z.h(0.5)
        };

        let e_log_v_q = if v == DEFAULT_V {
            1.0
        } else {
            z.pow_negative_q(v)
        };

        z.hxm = z.h(k + 0.5);
        z.hx0_minus_hxm = (h0x5 - e_log_v_q) - z.hxm; // h(0) = h(0.5) - exp(log(v) * -q)
        z.s = if use_precomputed {
            0.46153846153846123
        } else {
            1.0 - z.hinv(z.h(1.5) - z.pow_negative_q(v + 1.0))
        };

        z
    }

    fn h(&self, x: f64) -> f64 {
        if self.one_minus_q == DEFAULT_ONE_MINUS_Q {
            -1.0 / (self.v + x) // -exp(-log(v + x))
        } else {
            ((self.v + x).ln() * self.one_minus_q).exp() * self.one_minus_q_inv
        }
    }

    fn hinv(&self, x: f64) -> f64 {
        if self.one_minus_q == DEFAULT_ONE_MINUS_Q {
            -self.v - 1.0 / x // -v + exp(-log(-x))
        } else {
            -self.v + ((self.one_minus_q * x).ln() * self.one_minus_q_inv).exp()
        }
    }

    fn pow_negative_q(&self, x: f64) -> f64 {
        if self.q == DEFAULT_Q {
            1.0 / (x * x)
        } else {
            (x.ln() * -self.q).exp()
        }
    }

    #[cfg(test)]
    #[expect(clippy::many_single_char_names)]
    fn sample(&self, rng: &mut Rand<impl RandCore>) -> u64 {
        let mut k: f64;
        loop {
            let v = rng.next_float();
            let u = self.hx0_minus_hxm.mul_add(v, self.hxm);
            let x = self.hinv(u);
            k = x.round_ties_even();
            if k - x <= self.s {
                break;
            }
            let h = self.h(k + 0.5);
            let r = self.pow_negative_q(self.v + k);
            if u >= h - r {
                break;
            }
        }
        debug_assert!(k <= self.k);
        k as u64
    }

    fn sample_oneshot(&self, v: f64) -> u64 {
        let u = self.hx0_minus_hxm.mul_add(v, self.hxm);
        let x = self.hinv(u);
        let k = (x + 0.5) as u64;
        debug_assert!(k as f64 <= self.k);
        k
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        MAX_SIZE, TEMPERATURE_DEFAULT,
        rand::{DefaultRand, Sfc64, random_seed},
    };

    use super::*;

    #[test]
    fn geom_mean_approx() {
        let mut rng = DefaultRand::new(0);
        let n = 10_000;
        for mean in 0..=128 {
            let mut total = 0;
            let g = Geometric::new_with_mean(f64::from(mean));
            for _ in 0..n {
                total += g.sample(&mut rng);
            }
            let m = (total as f64) / f64::from(n);
            assert!(
                (m - f64::from(mean)).abs() / f64::from(mean.max(1)) < 0.1,
                "got mean {m} instead of {mean} with {n} values"
            );
        }
    }

    #[test]
    fn check_temperature_scale_down() {
        let t = temperature_scale_down(TEMPERATURE_DEFAULT / 2);
        assert_eq!(t, 0);
        let t = temperature_scale_down(TEMPERATURE_DEFAULT);
        assert_eq!(t, 0);
        let t = temperature_scale_down(TEMPERATURE_DEFAULT + TEMPERATURE_DEFAULT / 2);
        assert_eq!(t, 128);
        let t = temperature_scale_down(u8::MAX);
        assert_eq!(t, 254);
    }

    #[test]
    #[expect(clippy::float_cmp)]
    fn v_default_temperature() {
        let v = Biased::v_for_temperature(TEMPERATURE_DEFAULT);
        assert_eq!(v, Biased::V_DEFAULT);
        let v = Biased::v_for_temperature(TEMPERATURE_DEFAULT / 8);
        assert_eq!(v, DEFAULT_V);
        let v = Biased::v_for_temperature(0);
        assert_eq!(v, DEFAULT_V);
    }

    #[test]
    fn zipf_01_prob() {
        let mut rng = DefaultRand::new(0);
        let mut p0 = [0.0; 32];
        for (v, p) in p0.iter_mut().enumerate().skip(1) {
            let z = Zipf::new(1, Some(v as f64), None);
            let total = 10_000;
            let mut n0 = 0;
            for _ in 0..total {
                let ix = z.sample(&mut rng);
                if ix == 0 {
                    n0 += 1;
                }
            }
            *p = f64::from(n0) / f64::from(total);
        }
        for (i, p) in p0.into_iter().enumerate() {
            let p = p * 100.0;
            let c = (i as f64).mul_add(-0.065, 0.85) * 100.0;
            println!("{i:>3} {p:>5.1}% (calc {c:>5.1}%)");
        }
    }

    #[test]
    fn biased_stats() {
        let mut rng = DefaultRand::new(random_seed());
        for bound in [16, 32, 64, MAX_SIZE] {
            for v in [1.0, 2.0, 4.0, 8.0] {
                for q in [2.0] {
                    let biased = Biased::new(v, q);
                    let n = 100_000;
                    let mut s = 0;
                    let mut m = 0;
                    for _ in 0..n {
                        let i = biased.sample(&mut rng, bound);
                        m = m.max(i);
                        s += i;
                    }
                    let sz = (s as f64) / f64::from(n);
                    println!("average size for v={v}/q={q}/{bound}: {sz:.1} (max {m})");
                }
            }
        }
    }

    #[test]
    fn size_histogram() {
        let mut rng = DefaultRand::new(0);
        for bits in [1, 2, 3, 4, 8, 16, 32, 64, 128, 4096, 64 * 1024, 1024 * 1024] {
            let g12 = Geometric::new_with_mean(1.0 + (bits as f64).sqrt());
            let z1 = Zipf::new(bits - 1, None, None);
            let z2 = Zipf::new(bits - 1, Some(2.0), None);
            let z4 = Zipf::new(bits - 1, Some(4.0), None);
            let z8 = Zipf::new(bits - 1, Some(8.0), None);
            let z16 = Zipf::new(bits - 1, Some(16.0), None);
            let b0 = Biased::new_temperature(0, None);
            let b8 = Biased::new(8.0, DEFAULT_Q);
            #[expect(clippy::type_complexity)]
            let distribs: &[(&'static str, &dyn Fn(&mut DefaultRand) -> u64)] = &[
                ("g12", &|rng| bound_wrap(g12.sample(rng), bits - 1)),
                ("z1", &|rng| z1.sample(rng)),
                ("z2", &|rng| z2.sample(rng)),
                ("z4", &|rng| z4.sample(rng)),
                ("z8", &|rng| z8.sample(rng)),
                ("z16", &|rng| z16.sample(rng)),
                ("z8_oneshot", &|rng| z8.sample_oneshot(rng.next_float())),
                ("b0", &|rng| b0.sample(rng, bits as usize) as u64),
                ("b8", &|rng| b8.sample(rng, bits as usize) as u64),
            ];
            for (name, func) in distribs {
                let total = 100_000;
                let mut samples = vec![0; bits as usize];
                let mut sum = 0;
                for _ in 0..total {
                    let ix = (func)(&mut rng) as usize;
                    samples[ix] += 1;
                    sum += ix;
                }
                println!("{name} {bits}:");
                let mut t = 0usize;
                let mut perc = [(-1, 50), (-1, 75), (-1, 99)];
                for (i, s) in samples.into_iter().enumerate() {
                    t += s;
                    for (p, m) in &mut perc {
                        if *p == -1 && t >= total * *m / 100 {
                            *p = i as i32;
                        }
                    }
                    if bits <= 128 {
                        let h = "*".repeat(s / 1000);
                        let p = (s as f64) / (total as f64) * 100.0;
                        let q = (t as f64) / (total as f64) * 100.0;
                        println!("{i:>3}/{bits:>3} ({p:>5.1}% {q:>5.1}%) {h}");
                    }
                }
                let avg = (sum as f64) / (total as f64);
                println!("avg: {avg:.1}");
                for (p, m) in perc {
                    println!("p{m}: {p}");
                }
            }
        }
    }

    #[test]
    fn zipf_bound() {
        let mut rng = DefaultRand::new(0);
        for _ in 0..100 {
            let v = 1.0 - rng.next_float();
            let q = 2.0 - rng.next_float();
            let z = Zipf::new(100, Some(v), Some(q));
            for _ in 0..1000 {
                let u = z.sample(&mut rng);
                assert!(u <= 100);
                let u = z.sample_oneshot(rng.next_float());
                assert!(u <= 100);
            }
        }
    }

    #[test]
    fn zipf_zero() {
        let mut rng = DefaultRand::new(0);
        let z = Zipf::new(0, None, None);
        for _ in 0..10_000 {
            let u = z.sample(&mut rng);
            assert_eq!(u, 0);
        }
    }

    #[test]
    fn zipf_golden_default() {
        // reference: https://github.com/flyingmutant/rand/blob/master/std_zipf.go
        let golden: [u64; 20] = [0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 2, 0, 2, 12, 0, 1, 1, 0, 0, 1];
        let mut rng = Rand::<Sfc64>::new(0);
        let z = Zipf::new(100, None, None);
        for v in golden {
            let u = z.sample(&mut rng);
            assert_eq!(v, u);
        }
    }

    #[test]
    fn zipf_golden_custom() {
        // reference: https://github.com/flyingmutant/rand/blob/master/std_zipf.go
        let golden: [u64; 20] = [
            5, 1, 0, 12, 1, 8, 4, 0, 39, 6, 29, 2, 28, 76, 6, 19, 19, 8, 4, 10,
        ];
        let mut rng = Rand::<Sfc64>::new(0);
        let z = Zipf::new(100, Some(1.1), Some(1.1));
        for v in golden {
            let u = z.sample(&mut rng);
            assert_eq!(v, u);
        }
    }

    #[test]
    fn zipf_golden_custom_sub_1() {
        // reference: https://github.com/flyingmutant/rand/blob/master/std_zipf.go
        let golden: [u64; 20] = [
            3, 0, 0, 10, 0, 6, 3, 0, 37, 4, 26, 1, 25, 75, 4, 16, 16, 6, 2, 7,
        ];
        let mut rng = Rand::<Sfc64>::new(0);
        let z = Zipf::new(100, Some(0.5), Some(1.01));
        for v in golden {
            let u = z.sample(&mut rng);
            assert_eq!(v, u);
        }
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use crate::rand::DefaultRand;

    use super::*;
    use core::hint::black_box;

    #[bench]
    fn biased(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        let z = Biased::new(black_box(DEFAULT_V), DEFAULT_Q);
        b.iter(|| z.sample(&mut rng, black_box(7)));
    }

    #[bench]
    fn biased_bool(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        let z = Biased::new(black_box(4.0), DEFAULT_Q);
        b.iter(|| z.sample(&mut rng, black_box(2)));
    }

    #[bench]
    fn geometric(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        let z = Geometric::new_with_mean(black_box(100.0));
        b.iter(|| z.sample(&mut rng));
    }

    #[bench]
    fn zipf_new_v(b: &mut test::Bencher) {
        b.iter(|| Zipf::new(black_box(100), black_box(Some(1.1)), None));
    }

    #[bench]
    fn zipf_default(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        let z = Zipf::new(black_box(100), None, None);
        b.iter(|| z.sample(&mut rng));
    }

    #[bench]
    fn zipf_default_oneshot(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        let z = Zipf::new(black_box(100), None, None);
        b.iter(|| z.sample_oneshot(rng.next_float()));
    }

    #[bench]
    fn zipf_custom_v(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        let z = Zipf::new(black_box(100), black_box(Some(1.1)), None);
        b.iter(|| z.sample(&mut rng));
    }

    #[bench]
    fn zipf_custom_v_oneshot(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        let z = Zipf::new(black_box(100), black_box(Some(1.1)), None);
        b.iter(|| z.sample_oneshot(rng.next_float()));
    }

    #[bench]
    fn zipf_custom(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        let z = Zipf::new(black_box(100), black_box(Some(1.1)), black_box(Some(1.1)));
        b.iter(|| z.sample(&mut rng));
    }
}
