// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::math::{fast_reduce_u64, wide_mul};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Rand<R: RandCore> {
    rng: R,
}

pub(crate) type DefaultRand = Rand<Wyrand>;

impl<R: RandCore> Rand<R> {
    pub(crate) fn new(seed: u64) -> Self {
        Self { rng: R::new(seed) }
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.rng.next()
    }

    pub(crate) fn next_float(&mut self) -> f64 {
        low_53_to_float(self.next())
    }

    pub(crate) fn next_below(&mut self, n: usize) -> usize {
        self.next_below_u64(n as u64) as usize
    }

    pub(crate) fn next_below_u64(&mut self, n: u64) -> u64 {
        fast_reduce_u64(self.next(), n)
    }

    pub(crate) fn coinflip_fair(&mut self) -> bool {
        self.coinflip(0.5)
    }

    pub(crate) fn coinflip(&mut self, p: f64) -> bool {
        self.next_float() < p
    }
}

pub(crate) trait RandCore {
    fn new(seed: u64) -> Self;
    fn next(&mut self) -> u64;
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Wyrand {
    seed: u64,
}

impl Wyrand {
    pub(crate) fn mix(a: u64, b: u64) -> u64 {
        let (lo, hi) = wide_mul(a.wrapping_add(0x2d358dccaa6c78a5), b ^ 0x8bb84b93962eacc9);
        lo ^ hi
    }
}

impl RandCore for Wyrand {
    fn new(mut seed: u64) -> Self {
        Self {
            seed: splitmix64(&mut seed),
        }
    }

    fn next(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(0x2d358dccaa6c78a5);
        let (lo, hi) = wide_mul(self.seed, self.seed ^ 0x8bb84b93962eacc9);
        lo ^ hi
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Sfc64 {
    a: u64,
    b: u64,
    c: u64,
    w: u64,
}

impl RandCore for Sfc64 {
    fn new(seed: u64) -> Self {
        let mut s = Self {
            a: seed,
            b: seed,
            c: seed,
            w: 1,
        };
        for _ in 0..12 {
            s.next();
        }
        s
    }

    fn next(&mut self) -> u64 {
        let out = self.a.wrapping_add(self.b).wrapping_add(self.w);
        self.w = self.w.wrapping_add(1);
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = self.c.rotate_left(24).wrapping_add(out);
        out
    }
}

/// Return a pseudo-random 32-bit seed. For `no_std` environments, a single deterministic
/// sequence (controllable by [`jump_seed_sequence`]) is used.
#[must_use]
pub(crate) fn random_seed_32() -> u32 {
    let u = random_seed();
    ((u >> 32) ^ u) as u32
}

/// Return a pseudo-random 64-bit seed. For `no_std` environments, a single deterministic
/// sequence (controllable by [`jump_seed_sequence`]) is used.
#[must_use]
pub(crate) fn random_seed() -> u64 {
    use core::hash::{BuildHasher as _, Hasher as _};
    use std::collections::hash_map::RandomState;
    RandomState::new().build_hasher().finish()
}

// https://prng.di.unimi.it/splitmix64.c
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn low_53_to_float(n: u64) -> f64 {
    const MAX: u64 = 1u64 << 53;
    const MAX_DIV: f64 = 1.0 / (MAX as f64);
    let u = n & MAX.wrapping_sub(1);
    u as f64 * MAX_DIV
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wyrand_golden() {
        // https://github.com/wangyi-fudan/wyhash/blob/46cebe9dc4e51f94d0dca287733bc5a94f76a10d/wyhash.h#L150-L151
        let golden: [u64; 16] = [
            0x2aacd33c4686244c,
            0xb48f96f860a4684c,
            0x796a8be35b8e2cc5,
            0xe812184cc5b16f79,
            0x058ca39cd6d8809d,
            0x80495e6702b92698,
            0x9508a2bf9a450de4,
            0xbec2f0a83be18efe,
            0x04289dfd9f9e055c,
            0x3d146c0d65ff37d1,
            0x7c128413b0b4f341,
            0x104eb67899d77cd2,
            0x31ec81383ba9efbd,
            0xd6a197ea45e18065,
            0xee75a6e4be28c4a9,
            0x5398d90f8d88cdbe,
        ];

        let mut s = Wyrand::new(0);
        for v in golden {
            let u = s.next();
            assert_eq!(v, u);
        }
    }

    #[test]
    fn sfc64_golden() {
        // https://github.com/ziglang/zig/blob/25f1526fe6424cef156724977b75a5b80a3d5833/lib/std/Random/Sfc64.zig#L73-L99
        let golden: [u64; 16] = [
            0x3acfa029e3cc6041,
            0xf5b6515bf2ee419c,
            0x1259635894a29b61,
            0xb6ae75395f8ebd6,
            0x225622285ce302e2,
            0x520d28611395cb21,
            0xdb909c818901599d,
            0x8ffd195365216f57,
            0xe8c4ad5e258ac04a,
            0x8f8ef2c89fdb63ca,
            0xf9865b01d98d8e2f,
            0x46555871a65d08ba,
            0x66868677c6298fcd,
            0x2ce15a7e6329f57d,
            0xb2f1833ca91ca79,
            0x4b0890ac9bf453ca,
        ];

        let mut s = Sfc64::new(0);
        for v in golden {
            let u = s.next();
            assert_eq!(v, u);
        }
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use super::*;
    use core::hint::black_box;

    #[bench]
    fn splitmix(b: &mut test::Bencher) {
        let mut v = black_box(0);
        b.iter(|| splitmix64(&mut v));
    }

    #[bench]
    fn default_next(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        b.iter(|| rng.next());
    }

    #[bench]
    fn default_next_float(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        b.iter(|| rng.next_float());
    }

    #[bench]
    fn default_next_below(b: &mut test::Bencher) {
        let mut rng = DefaultRand::new(black_box(0));
        b.iter(|| rng.next_below(black_box(100)));
    }

    #[bench]
    fn sfc64_next(b: &mut test::Bencher) {
        let mut rng = Rand::<Sfc64>::new(black_box(0));
        b.iter(|| rng.next());
    }

    #[bench]
    fn sfc64_next_float(b: &mut test::Bencher) {
        let mut rng = Rand::<Sfc64>::new(black_box(0));
        b.iter(|| rng.next_float());
    }

    #[bench]
    fn sfc64_next_below(b: &mut test::Bencher) {
        let mut rng = Rand::<Sfc64>::new(black_box(0));
        b.iter(|| rng.next_below(black_box(100)));
    }
}
