// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Constant-time stateless permutation of [0, n).
pub(crate) fn permute(mut ix: usize, n: usize, seed: u64) -> usize {
    debug_assert!(ix < n);
    match n {
        0 => unreachable!(),
        1 => 0,
        _ => {
            // From "Correlated Multi-Jittered Sampling" by Andrew Kensler:
            // https://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf.
            // Simplified explanation: https://andrew-helmer.github.io/permute/.
            let mask = usize::MAX >> (n - 1).leading_zeros();
            loop {
                ix = hash(ix as u64, mask as u64, seed) as usize;
                if ix < n {
                    return ix;
                }
            }
        }
    }
}

fn hash(mut v: u64, mask: u64, seed: u64) -> u64 {
    // Based on rrxmrrxmsx_0 mixer by Pelle Evensen:
    // https://mostlymangling.blogspot.com/2019/01/better-stronger-mixer-and-test-procedure.html.
    v ^= seed;
    v ^= v.rotate_right(25) ^ v.rotate_right(50);
    v = v.wrapping_mul(0xa24baed4963ee407);
    v ^= (v & mask).rotate_right(24) ^ (v & mask).rotate_right(49);
    v = v.wrapping_mul(0x9fb21c651e98df25);
    v & mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rand::random_seed;

    #[test]
    fn permute_dedup() {
        for _ in 0..128 {
            let m = 128;
            let mut permuted = Vec::with_capacity(m);
            for n in 1..m {
                let seed = random_seed();
                permuted.clear();
                for ix in 0..n {
                    let p = permute(ix, n, seed);
                    assert!(p < n);
                    permuted.push(p);
                }
                permuted.sort_unstable();
                permuted.dedup();
                assert_eq!(permuted.len(), n);
            }
        }
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use super::*;
    use crate::rand::{DefaultRand, random_seed};
    use core::hint::black_box;

    #[bench]
    fn permute_(b: &mut test::Bencher) {
        let seed = random_seed();
        let mut rng = DefaultRand::new(seed);
        b.iter(|| {
            let n = 100_500;
            let ix = rng.next_below(n);
            permute(black_box(ix), black_box(n), black_box(seed))
        });
    }
}
