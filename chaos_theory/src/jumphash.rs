// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// J. Lamping, E. Veach: "A Fast, Minimal Memory, Consistent Hash Algorithm".
#[expect(clippy::allow_attributes)]
#[allow(dead_code, reason = "will be used in new stable swarm algorithms")]
pub(crate) fn jumphash(seed: u64, buckets: u64) -> u64 {
    let mut r = seed;
    let mut b = 0u64;
    let mut j = 0u64;
    while j < buckets {
        b = j;
        r = r.wrapping_mul(2862933555777941757).wrapping_add(1);
        // Integer version of `floor((b + 1) / r)`.
        if buckets <= (1u64 << 33) {
            j = ((b + 1) << 31) / ((r >> 33) + 1);
        } else {
            j = ((u128::from(b + 1) << 31) / u128::from((r >> 33) + 1)).min(u64::MAX.into()) as u64;
        }
    }
    debug_assert!(b < buckets);
    b
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use super::*;
    use crate::rand::random_seed;
    use core::hint::black_box;

    #[bench]
    fn jumphash_small(b: &mut test::Bencher) {
        let seed = random_seed();
        b.iter(|| jumphash(black_box(seed), black_box(u16::MAX.into())));
    }

    #[bench]
    fn jumphash_big(b: &mut test::Bencher) {
        let seed = random_seed();
        b.iter(|| jumphash(black_box(seed), black_box(u64::MAX)));
    }
}
