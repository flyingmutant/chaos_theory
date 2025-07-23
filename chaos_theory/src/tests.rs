// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{cmp::Ordering, mem::swap};
use std::collections::{HashMap, HashSet};

use crate::{
    Arbitrary as _, BUDGET_DEFAULT, CHECK_ITERS_DEFAULT, Effect, Env, Generator, Set, Source,
    TEMPERATURE_DEFAULT, assume, check, make,
    rand::{DefaultRand, random_seed},
    slow_test_enabled,
    tape::Tape,
    tape_mutate_crossover::CrossoverCache,
    unwind, vprintln,
};

const NUM_EXAMPLES: usize = CHECK_ITERS_DEFAULT;

// TODO: make public
pub(crate) fn print_debug_examples<G: Generator>(
    g: G,
    like: Option<&G::Item>,
    cmp: impl FnMut(&G::Item, &G::Item) -> Ordering,
) {
    println!("{g:#?}:");
    let mut items = Vec::with_capacity(NUM_EXAMPLES);
    let mut env = Env::new();
    for _ in 0..NUM_EXAMPLES {
        let v = env.example_of(&g, like);
        items.push(v);
    }
    items.sort_by(cmp);
    println!("{items:#?}");
}

pub(crate) fn any_assert_valid_tape<G: Generator>(
    src: &mut Source,
    label: &'static str,
    g: G,
) -> G::Item {
    let chk = src.as_raw().as_mut().tape_checkpoint();
    let v = src.any_of(label, g);
    let tape = src
        .as_raw()
        .as_mut()
        .tape_copy_from_checkpoint(chk, true, true);
    tape.debug_assert_valid();
    v
}

// TODO: make public
pub(crate) fn prop_smoke(
    src: &mut Source,
    label: &'static str,
    g: impl Generator<Item: PartialEq>,
) {
    src.scope(label, |src| {
        let chk = src.as_raw().as_mut().tape_checkpoint();
        // Ensure that the generation works, in general.
        let example = src.any_of("example", &g);
        let tape = src
            .as_raw()
            .as_mut()
            .tape_copy_from_checkpoint(chk, true, true);

        // Ensure the generator tape is valid.
        tape.debug_assert_valid();
        let value = src.as_raw().any_of("value", &g, Some(&example));

        // Ensure generator can reconstruct examples.
        assert_eq!(value, example);

        // Ensure that after discarding noop data, we generate the same value.
        let tape_ = tape.discard_noop();
        let mut env = src.as_raw().derived_oneshot_env(tape_);
        let example_ = env.example_of(g, None);
        assert_eq!(example_, value);
    });
}

#[cfg(feature = "_bench")]
pub(crate) fn bench_gen_next(b: &mut test::Bencher, g: &impl Generator) {
    let mut env = Env::custom().with_rng_budget(usize::MAX).env(true);
    let mut src = env.__start_from_nothing(true);
    let g = core::hint::black_box(g);
    let src = core::hint::black_box(src.as_raw());
    let example = core::hint::black_box(None);
    b.iter(|| g.next(src, example));
}

// TODO: single prop that depends on the values in noop scopes + test that we correctly minify it

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct RgbState {
    r: u8,
    g: u8,
    b: u8,
}

impl RgbState {
    // TODO: use chaos_theory to construct a truly random property!
    // TODO: use signed integers here as well
    // TODO: use all our primitives and generators here
    // TODO: fail the test not only at the top-level repeat
    pub(crate) fn prop_fill(&mut self, src: &mut Source) {
        let seeds_a = [0, 1, 2, 3, 4, 5];
        let seeds_b = [vec![0, 1], vec![2, 3], vec![7]];
        let seeds_c = [vec![vec![3, 1], vec![0, 2]], vec![vec![5]]];

        let amount_gen = make::int_in_range(1..)
            .seeded(&seeds_a, true)
            .filter_assume(|a| *a != 3)
            .collect_n::<Vec<_>>(1..3)
            .seeded(&seeds_b, true)
            .collect_n::<Vec<Vec<_>>>(1..3)
            .seeded(&seeds_c, true)
            .map(|mut v| v.pop().unwrap())
            .seeded(&seeds_b, true);

        src.repeat_select("target", &["r", "g", "b"], |src, variant, _ix| {
            assert_ne!(self.r.max(self.g).max(self.b), u8::MAX);
            match variant {
                "r" => {
                    if self.r > self.g.max(self.b) {
                        return Effect::Noop;
                    }
                    if src.any("swap g") {
                        swap(&mut self.r, &mut self.g);
                    }
                    if src.any("swap b") {
                        swap(&mut self.r, &mut self.b);
                    }
                    Effect::Success
                }
                "g" => {
                    if self.g > self.r.max(self.b) {
                        return Effect::Noop;
                    }
                    // TODO: how to observe the coverage in cases like these?
                    src.select("action", &["up", "down"], |src, variant, _ix| {
                        match variant {
                            "up" => {
                                if self.g == u8::MAX {
                                    return Effect::Noop;
                                }
                                self.g += 1;
                            }
                            "down" => {
                                let (&div, _) = src.choose("div", &[0, 1, 2]).unwrap();
                                if div == 0 {
                                    return Effect::Noop;
                                }
                                self.g /= div;
                            }
                            _ => unreachable!(),
                        }
                        Effect::Success
                    })
                }
                "b" => {
                    if self.b > self.r.max(self.g) {
                        return Effect::Noop;
                    }
                    if self.b < self.r.min(self.g) {
                        src.maybe("boost", |src| {
                            let (&mult, _) = src.choose("mult", &[1, 2, 3, 4, 5]).unwrap();
                            self.b = self.b.saturating_mul(mult);
                        });
                    } else {
                        // Add nested find that sometimes is failing.
                        let ignore = src.find("ignore", |src| {
                            let v: u8 = src.any("v");
                            (v % 17 == 1).then_some(v)
                        });
                        // Sometimes exit immediately after unsuccessful find, sometimes make one more draw.
                        if ignore.is_none() && self.b % 2 == 1 && src.any("early_noop") {
                            return Effect::Noop;
                        }
                        let mut total: u8 = 0;
                        src.repeat("increase", |src| {
                            // Get a value, but in a relatively convoluted way, to touch more of the code.
                            let amount: u8 = *src
                                .any_of("amount", &amount_gen)
                                .iter()
                                .min()
                                .expect("vector is not empty");
                            if amount % 2 == 0 {
                                return Effect::Noop;
                            }
                            total = total.saturating_add(amount);
                            Effect::Success
                        });
                        if total % 2 == 0 {
                            return Effect::Noop;
                        }
                        self.b = self.b.saturating_add(total);
                    }
                    Effect::Success
                }
                _ => unreachable!(),
            }
        });
    }

    // Note, without `assume_no_err`, any draw from this Source after `prop_fill_tape` can lead to an invalid tape
    // (you can't append events to a tape that was terminated in the middle of something by prop failure).
    #[track_caller]
    pub(crate) fn prop_fill_tape(
        &mut self,
        src: &mut Source,
        assume_no_err: bool,
        fill_choices: bool,
        copy_meta: bool,
    ) -> Tape {
        let chk = src.as_raw().as_mut().tape_checkpoint();
        let r = unwind::catch_silent(
            &mut |src| {
                self.prop_fill(src);
            },
            src,
        );
        if assume_no_err {
            assume!(r.is_ok());
        } else if let Err(e) = r {
            assume!(!e.invalid_data);
        }
        let tape = src
            .as_raw()
            .as_mut()
            .tape_copy_from_checkpoint(chk, fill_choices, copy_meta);
        tape.debug_assert_valid();
        tape
    }

    #[track_caller]
    pub(crate) fn prop_replay_from_tape(&mut self, src: &mut Source, tape: Tape) {
        tape.debug_assert_valid();
        let mut env = src.as_raw().derived_oneshot_env(tape);
        let _ = env.check_silent(|src| {
            *self = Self::default();
            self.prop_fill(src);
        });
        assert!(!env.rng_used());
    }
}

fn print_crossover_examples<G: Generator<Item: Ord>>(
    g: &G,
    example1: &G::Item,
    example2: &G::Item,
) {
    let tape1 = Env::produce_tape(0, TEMPERATURE_DEFAULT, BUDGET_DEFAULT, |src| {
        let _ = src.as_raw().any_of("example", &g, Some(example1));
    })
    .unwrap();
    let tape2 = Env::produce_tape(0, TEMPERATURE_DEFAULT, BUDGET_DEFAULT, |src| {
        let _ = src.as_raw().any_of("example", &g, Some(example2));
    })
    .unwrap();

    let mut rng = DefaultRand::new(random_seed());
    let mut all_examples = Vec::with_capacity(NUM_EXAMPLES);
    let mut cache = CrossoverCache::default();
    for _ in 0..NUM_EXAMPLES / 2 {
        let cross1 = tape1.clone().crossover(
            &tape2,
            &mut rng,
            TEMPERATURE_DEFAULT,
            true,
            true,
            &mut cache,
        );
        let cross2 = tape2.clone().crossover(
            &tape1,
            &mut rng,
            TEMPERATURE_DEFAULT,
            true,
            true,
            &mut cache,
        );
        for cross in [cross1, cross2] {
            let mut env = Env::custom().with_rng_tape(cross).env(false);
            let example = env.example_of(&g, None);
            all_examples.push(example);
        }
    }
    all_examples.sort();
    println!("{all_examples:#?}");
}

#[test]
fn crossover_examples_vec_strings() {
    let g = make::arbitrary::<String>().collect::<Vec<_>>();
    print_crossover_examples(
        &g,
        &vec![
            "HELLO,".to_owned(),
            "WORLD!".to_owned(),
            "AND:".to_owned(),
            "GOODBYE :-)".to_owned(),
        ],
        &vec![
            "to".to_owned(),
            "hell".to_owned(),
            "with".to_owned(),
            "all".to_owned(),
            "of".to_owned(),
            "you".to_owned(),
        ],
    );
}

#[test]
fn crossover_examples_vec_u32() {
    let g = make::arbitrary::<u32>().collect::<Vec<_>>();
    print_crossover_examples(&g, &(1..5).collect(), &(100..110).collect());
}

#[test]
#[should_panic(expected = "assertion failed")]
fn generate_same() {
    Env::custom()
        .with_check_iters(64 * 256)
        .env(true)
        .check(|src| {
            let mut objs = Set::default();
            src.repeat("step", |src| {
                let obj: Vec<u8> = src.any("obj");
                if obj.len() > 3 && obj.iter().any(|v| *v != 0) {
                    let unique = objs.insert(obj.clone());
                    if !unique {
                        vprintln!(src, "duplicate: {obj:?}");
                    }
                    assert!(unique);
                }
                Effect::Success
            });
        });
}

#[test]
fn vec_like_examples() {
    print_debug_examples(
        &make::arbitrary::<Vec<(u32, u32, u32)>>(),
        Some(&vec![(111, 111, 111), (222, 222, 222), (333, 333, 333)]),
        |a, b| (a.len(), a).cmp(&(b.len(), b)),
    );
}

#[test]
#[should_panic(expected = "assertion `left != right` failed")]
fn fail_rgb_fill() {
    check(|src| {
        let mut s = RgbState::default();
        s.prop_fill(src);
    });
}

#[test]
#[should_panic(expected = "assumption failed")]
fn fail_assume_trivial() {
    check(|src| {
        let i: i32 = src.any("i");
        assume!(i == 123_456_789);
    });
}

#[test]
fn test_assume_working() {
    check(|src| {
        let i: i32 = src.any("i");
        assume!(i != 0);
        assert_ne!(i, 0);
    });
}

#[test]
#[should_panic(expected = "assumption failed")]
fn fail_assume_filter() {
    check(|src| {
        let _never = src.any_of("never", make::just(0).filter_assume(|i| *i != 0));
    });
}

#[test]
#[should_panic(expected = "assertion `left != right` failed")]
fn fail_trivial_int() {
    check(|src| {
        let i: i32 = src.any("i");
        assert_ne!(i, 0);
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
fn fail_trivial_float() {
    check(|src| {
        let f: f32 = src.any("f");
        assert!(f.is_nan());
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
fn fail_trivial_vec() {
    check(|src| {
        let v: Vec<i32> = src.any("i");
        assert!(v.is_empty());
    });
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn fail_trivial_repeat() {
    check(|src| {
        let mut n: i32 = 0;
        src.repeat("add", |src| {
            let a = src.any("a");
            n = n.wrapping_add(a);
            Effect::Success
        });
        assert_eq!(n, 0);
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
fn fail_repeat() {
    check(|src| {
        let mut v = Vec::new();
        // Note: repeat + select instead of repeat_select.
        src.repeat("rep2", |src| {
            src.select(
                "step",
                &["a", "b", "c"],
                |src, variant, _ix| match variant {
                    "a" => {
                        v.push(src.any("a_int"));
                    }
                    "b" => {
                        v.push(2);
                    }
                    "c" => {
                        v.push(3);
                    }
                    _ => unreachable!(),
                },
            );
            Effect::Success
        });
        assert!(v.is_empty());
    });
}

#[test]
#[should_panic(expected = "oops")]
fn fail_repeat_select_n() {
    check(|src| {
        src.repeat_select_n(
            "action",
            1..=1,
            &["nop", "fail"],
            |_src, var, _var_ix| match var {
                "nop" => Effect::Noop,
                "fail" => {
                    panic!("oops")
                }
                _ => unreachable!(),
            },
        );
    });
}

#[test]
#[should_panic(expected = "assertion failed")]
fn fail_log_debug() {
    check(|src| {
        let b: bool = src.any("b");
        let _i: i32 = src.any("i");
        let _o: Option<usize> = src.any("o");
        let _r: Result<isize, bool> = src.any("r");
        let _v: Vec<i32> = src.any("v");
        let _s: HashSet<i8> = src.any("s");
        let _m: HashMap<u8, u64> = src.any("m");
        let _si = src.choose("si", &["a", "b", "c"]);
        let _sb = {
            src.select("sb", &["br_a", "br_b"], |src, variant, _ix| match variant {
                "br_a" => src.any_of("a", make::just("AAA")),
                "br_b" => src.any_of("b", make::just("BBB")),
                _ => unreachable!(),
            })
        };
        assert!(b);
    });
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn fail_crackaddr() {
    // Based on https://gist.github.com/withzombies/96fa5e69b668fc9fa3c4bd6db5ef93fc
    fn crackaddr(input: &[u8], buf: &mut [u8]) {
        let mut d = 0;
        let mut upper = buf.len() / 2;
        let mut quotation = false;
        let mut round_quote = false;

        for &c in input {
            if c == b'<' && !quotation {
                quotation = true;
                upper -= 1;
            }
            if c == b'>' && quotation {
                quotation = false;
                upper += 1;
            }
            if c == b'(' && !quotation && !round_quote {
                round_quote = true;
                // upper -= 1;
            }
            if c == b')' && !quotation && round_quote {
                round_quote = false;
                upper += 1;
            }
            if d < upper {
                buf[d] = c;
                d += 1;
            }
        }
        if round_quote {
            buf[d] = b')';
            d += 1;
        }
        if quotation {
            buf[d] = b'>';
        }
    }

    // Immediately panic with desired message if slow tests are disabled.
    assert!(slow_test_enabled(), "index out of bounds");

    let g = u8::arbitrary().seeded(b"<>()", false).collect::<Vec<_>>();

    Env::custom()
        .with_check_iters(1024 * 1024)
        .env(true)
        .check(|src| {
            let input: Vec<u8> = src.any_of("input", &g);
            let mut buf = vec![0; 16]; // originally 64
            crackaddr(&input, &mut buf);
        });
}
