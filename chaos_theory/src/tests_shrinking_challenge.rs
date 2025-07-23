// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{Env, Generator as _, Set, check, make};

#[test]
#[should_panic(expected = "assertion failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/bound5.md
fn fail_bound5() {
    fn sum(v: &[i16]) -> i16 {
        let mut i: i16 = 0;
        for n in v {
            i = i.wrapping_add(*n);
        }
        i
    }

    check(|src| {
        const C: i16 = 256;
        let t: Vec<Vec<i16>> = src.any_of(
            "t",
            make::arbitrary::<i16>()
                .collect::<Vec<_>>()
                .filter_assume(|v| sum(v) < C)
                .collect_n::<Vec<Vec<_>>>(5..=5),
        );

        let mut s: i16 = 0;
        for v in &t {
            s = s.wrapping_add(sum(v));
        }
        assert!(s < t.len() as i16 * C);
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/large_union_list.md
fn fail_large_union_list() {
    check(|src| {
        let v: Vec<Vec<i32>> = src.any_of(
            "v",
            make::arbitrary::<i32>()
                .collect::<Vec<_>>()
                .collect::<Vec<Vec<_>>>(),
        );
        assert!(v.iter().flatten().collect::<Set<_>>().len() <= 4);
    });
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/reverse.md
fn fail_reverse() {
    check(|src| {
        let v: Vec<i32> = src.any("v");
        let mut r = v.clone();
        r.reverse();
        assert_eq!(v, r);
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/lengthlist.md
fn fail_length_list() {
    check(|src| {
        let v = src.any_of(
            "v",
            make::int_in_range(1..=100)
                .and_then(|n| ("list", make::arbitrary::<i32>().collect_n::<Vec<_>>(n..=n))),
        );
        assert!(v.iter().max().copied().unwrap_or_default() < 900);
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/difference.md
fn fail_difference_not_zero() {
    Env::custom()
        .with_check_iters(64 * 256)
        .env(true)
        .check(|src| {
            let a: i32 = src.any("a");
            let b: i32 = src.any("b");
            assert!(a < 10 || a.saturating_sub(b) != 0);
        });
}

#[test]
#[should_panic(expected = "assertion failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/difference.md
fn fail_difference_small() {
    Env::custom()
        .with_check_iters(64 * 256)
        .env(true)
        .check(|src| {
            let a: i32 = src.any("a");
            let b: i32 = src.any("b");
            let d = a.saturating_sub(b).wrapping_abs();
            assert!(a < 10 || !(1..=4).contains(&d));
        });
}

#[test]
#[should_panic(expected = "assertion failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/difference.md
fn fail_difference_not_one() {
    Env::custom()
        .with_check_iters(256 * 256)
        .env(true)
        .check(|src| {
            let a: i32 = src.any("a");
            let b: i32 = src.any("b");
            assert!(a < 10 || a.saturating_sub(b).abs() != 1);
        });
}

#[test]
#[should_panic(expected = "assertion `left != right` failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/coupling.md
fn fail_coupling() {
    check(|src| {
        let v: Vec<_> = src.any_of(
            "v",
            make::int_in_range(0..=10)
                .collect::<Vec<_>>()
                .filter_assume(|v| v.iter().all(|i| *i < v.len())),
        );

        for (i, j) in v.iter().enumerate() {
            if i != *j {
                assert_ne!(v[*j], i);
            }
        }
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/deletion.md
fn fail_deletion() {
    check(|src| {
        let mut v: Vec<i32> = src.any_of(
            "v",
            make::arbitrary::<Vec<_>>().filter_assume(|v| !v.is_empty()),
        );
        let (&i, _) = src.choose("i", &v).unwrap();
        v.remove(v.iter().position(|n| *n == i).unwrap());
        assert!(v.iter().all(|n| *n != i));
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/distinct.md
fn fail_distinct() {
    check(|src| {
        let v: Vec<i32> = src.any("v");
        assert!(Set::from_iter(v).len() < 3);
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
// https://github.com/jlink/shrinking-challenge/blob/main/challenges/nestedlists.md
fn fail_nested_lists() {
    check(|src| {
        let v = src.any_of(
            "v",
            make::just(0).collect::<Vec<_>>().collect::<Vec<Vec<_>>>(),
        );
        assert!(v.iter().map(Vec::len).sum::<usize>() <= 10);
    });
}

// TODO: https://github.com/jlink/shrinking-challenge/blob/main/challenges/calculator.md
// TODO: https://github.com/jlink/shrinking-challenge/blob/main/challenges/binheap.md
