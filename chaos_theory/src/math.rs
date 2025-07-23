// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::Unsigned;

pub(crate) const fn percent(n: usize) -> f64 {
    debug_assert!(n <= 100);
    (n as f64) / 100.
}

pub(crate) fn wide_mul(a: u64, b: u64) -> (u64, u64) {
    let m = u128::from(a).wrapping_mul(u128::from(b));
    (m as u64, (m >> 64) as u64)
}

pub(crate) fn fast_reduce(r: u64, n: usize) -> usize {
    fast_reduce_u64(r, n as u64) as usize
}

pub(crate) fn fast_reduce_u64(r: u64, n: u64) -> u64 {
    debug_assert_ne!(n, 0);
    let (_, hi) = wide_mul(n, r);
    hi
}

pub(crate) fn bitmask<U: Unsigned>(bits: usize) -> U {
    if bits == 0 {
        U::ZERO
    } else {
        U::MAX >> (U::BITS - bits)
    }
}

#[cfg(test)]
mod tests {
    use crate::{check, make};

    use super::*;

    #[test]
    fn fast_reduce() {
        check(|src| {
            let r = src.any("r");
            let n = src.any_of("n", make::int_in_range(1..));
            let v = fast_reduce_u64(r, n);
            assert!(v < n);
        });
    }

    #[test]
    fn bitmask_bitlen() {
        check(|src| {
            let bits = src.any_of("bits", make::int_in_range(..=64usize));
            let m = bitmask::<u64>(bits);
            assert_eq!(m.bit_len(), bits);
            let u: u64 = src.any("u");
            let v = u & m;
            assert!(v.bit_len() <= m.bit_len());
        });
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use core::hint::black_box;

    #[bench]
    fn rint(b: &mut test::Bencher) {
        b.iter(|| black_box(123.456_f64).round_ties_even());
    }

    #[bench]
    fn log1p(b: &mut test::Bencher) {
        b.iter(|| black_box(123.456_f64).ln_1p());
    }
}
