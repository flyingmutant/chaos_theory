// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    cmp::{self, Ordering},
    fmt::Debug,
    ops::RangeBounds,
};

use jiff::{SignedDuration, Timestamp};

use crate::{Arbitrary, Generator, Ranged, SourceRaw, Tweak, make, math::percent, range::Range};

// Keep these choices aligned with `make_time` where possible. In particular, the
// `<secs>` and `<nanos>` scopes are intentionally the same so that tapes for standard
// and Jiff time types can be reused by mutation and crossover.
const DURATION_SPECIAL_PROB: f64 = percent(15);
const SECOND_SPECIALS: &[i64] = &[60, 3600, 86400];
const NANOSECOND_SPECIALS: &[i32] = &[1000, 1_000_000, 500_000_000];

const NANOS_PER_SEC: i32 = 1_000_000_000;
const MAX_SUBSEC_NANOS: i32 = NANOS_PER_SEC - 1;
const SMALLEST_DURATION: SignedDuration = SignedDuration::from_nanos(1);
const TIMESTAMP_ANCHOR: Timestamp = Timestamp::constant(946684800, 0); // 2000/01/01 00:00:00Z

#[cfg_attr(docsrs, doc(cfg(feature = "jiff")))]
impl Arbitrary for Timestamp {
    fn arbitrary() -> impl Generator<Item = Self> {
        timestamp_in_range(..)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "jiff")))]
impl Arbitrary for SignedDuration {
    fn arbitrary() -> impl Generator<Item = Self> {
        signed_duration_in_range(..)
    }
}

#[derive(Clone, Copy)]
struct SecNanosRange {
    secs: Range<i64>,
    nanos_start: i32,
    nanos_end: i32,
}

impl SecNanosRange {
    fn new(min_secs: i64, min_nanos: i32, max_secs: i64, max_nanos: i32) -> Self {
        Self {
            secs: Range::new_raw(min_secs, max_secs),
            nanos_start: min_nanos,
            nanos_end: max_nanos,
        }
    }

    fn nanos(&self, secs: i64) -> Range<i32> {
        let (mut min, mut max) = match secs.cmp(&0) {
            Ordering::Less => (-MAX_SUBSEC_NANOS, 0),
            Ordering::Equal => (-MAX_SUBSEC_NANOS, MAX_SUBSEC_NANOS),
            Ordering::Greater => (0, MAX_SUBSEC_NANOS),
        };
        if secs == self.secs.min {
            min = cmp::max(min, self.nanos_start);
        }
        if secs == self.secs.max {
            max = cmp::min(max, self.nanos_end);
        }
        Range::new_raw(min, max)
    }
}

fn sec_nanos_next(
    src: &mut SourceRaw,
    range: SecNanosRange,
    example: Option<(i64, i32)>,
) -> (i64, i32) {
    let mut example_secs = example.map(|(secs, _)| secs);
    if example_secs.is_none() {
        example_secs = src
            .as_mut()
            .choose_seed(DURATION_SPECIAL_PROB, SECOND_SPECIALS)
            .copied()
            .and_then(|secs| {
                if range.secs.contains(&secs) {
                    Some(secs)
                } else if range.secs.contains(&-secs) {
                    Some(-secs)
                } else {
                    None
                }
            });
    }
    let secs = src.any_of(
        "<secs>",
        make::int_in_range(range.secs),
        example_secs.as_ref(),
    );

    let nanos_range = range.nanos(secs);
    let mut example_nanos = example.map(|(_, nanos)| nanos);
    if example_nanos.is_none() {
        example_nanos = src
            .as_mut()
            .choose_seed(DURATION_SPECIAL_PROB, NANOSECOND_SPECIALS)
            .copied()
            .and_then(|nanos| {
                if nanos_range.contains(&nanos) {
                    Some(nanos)
                } else if nanos_range.contains(&-nanos) {
                    Some(-nanos)
                } else {
                    None
                }
            });
    }
    let nanos = src.any_of(
        "<nanos>",
        make::int_in_range(nanos_range),
        example_nanos.as_ref(),
    );

    (secs, nanos)
}

fn signed_duration_parts(duration: SignedDuration) -> (i64, i32) {
    (duration.as_secs(), duration.subsec_nanos())
}

fn signed_duration_sec_nanos(range: Range<SignedDuration>) -> SecNanosRange {
    let (min_secs, min_nanos) = signed_duration_parts(range.min);
    let (max_secs, max_nanos) = signed_duration_parts(range.max);
    SecNanosRange::new(min_secs, min_nanos, max_secs, max_nanos)
}

impl Generator for SignedDuration_ {
    type Item = SignedDuration;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example = example.copied().map(signed_duration_parts);
        let (secs, nanos) = sec_nanos_next(src, self.sec_nanos, example);
        SignedDuration::new(secs, nanos)
    }
}

struct SignedDuration_ {
    range: Range<SignedDuration>,
    sec_nanos: SecNanosRange,
}

impl Debug for SignedDuration_ {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SignedDuration").field(&self.range).finish()
    }
}

impl Ranged for SignedDuration {
    const ZERO: Self = Self::ZERO;
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;

    fn next_up(self) -> Option<Self> {
        self.checked_add(SMALLEST_DURATION)
    }

    fn next_down(self) -> Option<Self> {
        self.checked_sub(SMALLEST_DURATION)
    }
}

/// Create a generator of Jiff [`SignedDuration`] values in `range`.
#[cfg_attr(docsrs, doc(cfg(feature = "jiff")))]
pub fn signed_duration_in_range(
    range: impl RangeBounds<SignedDuration>,
) -> impl Generator<Item = SignedDuration> {
    let range = Range::new(range);
    SignedDuration_ {
        range,
        sec_nanos: signed_duration_sec_nanos(range),
    }
}

fn timestamp_parts(timestamp: Timestamp) -> (i64, i32) {
    (timestamp.as_second(), timestamp.subsec_nanosecond())
}

fn timestamp_sec_nanos(range: Range<Timestamp>) -> SecNanosRange {
    let (min_secs, min_nanos) = timestamp_parts(range.min);
    let (max_secs, max_nanos) = timestamp_parts(range.max);
    SecNanosRange::new(min_secs, min_nanos, max_secs, max_nanos)
}

impl Generator for Timestamp_ {
    type Item = Timestamp;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_before = example.map(|timestamp| *timestamp < TIMESTAMP_ANCHOR);
        let (example_before, forced) = match (self.before, self.after) {
            (Some(_), Some(_)) => (example_before, false),
            (Some(_), None) => (Some(true), true),
            (None, Some(_)) => (Some(false), true),
            (None, None) => unreachable!(),
        };
        if forced {
            src.mark_next_choice_forced();
        }
        let ix = src
            .as_mut()
            .choose_index(2, example_before.map(usize::from), Tweak::None);
        let range = if ix == 0 { &self.after } else { &self.before };
        let range = range.expect("internal error: range not set");
        let example = example.copied().map(timestamp_parts);
        let (secs, nanos) = sec_nanos_next(src, range, example);
        Timestamp::new(secs, nanos).expect("internal error: generated invalid Timestamp")
    }
}

struct Timestamp_ {
    range: Range<Timestamp>,
    before: Option<SecNanosRange>,
    after: Option<SecNanosRange>,
}

impl Debug for Timestamp_ {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Timestamp").field(&self.range).finish()
    }
}

impl Ranged for Timestamp {
    const ZERO: Self = Self::UNIX_EPOCH;
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;

    fn next_up(self) -> Option<Self> {
        self.checked_add(SMALLEST_DURATION).ok()
    }

    fn next_down(self) -> Option<Self> {
        self.checked_sub(SMALLEST_DURATION).ok()
    }
}

/// Create a generator of Jiff [`Timestamp`] values in `range`.
#[cfg_attr(docsrs, doc(cfg(feature = "jiff")))]
pub fn timestamp_in_range(range: impl RangeBounds<Timestamp>) -> impl Generator<Item = Timestamp> {
    let range = Range::new(range);
    let (before, after) = if range.min >= TIMESTAMP_ANCHOR {
        (None, Some(timestamp_sec_nanos(range)))
    } else if range.max < TIMESTAMP_ANCHOR {
        (Some(timestamp_sec_nanos(range)), None)
    } else {
        (
            Some(timestamp_sec_nanos(Range::new_raw(
                range.min,
                TIMESTAMP_ANCHOR,
            ))),
            Some(timestamp_sec_nanos(Range::new_raw(
                TIMESTAMP_ANCHOR,
                range.max,
            ))),
        )
    };
    Timestamp_ {
        range,
        before,
        after,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BUDGET_DEFAULT, Env, Source, TEMPERATURE_DEFAULT, check, make,
        tape::Tape,
        tests::{print_debug_examples, prop_smoke},
    };
    use core::time::Duration;
    #[cfg(feature = "std")]
    use std::time::UNIX_EPOCH;

    #[test]
    fn signed_duration_smoke() {
        check(|src| {
            prop_smoke(src, "SignedDuration", SignedDuration::arbitrary());
        });
    }

    #[test]
    fn signed_duration_gen_in_range() {
        check(|src| {
            let range: Range<SignedDuration> = src.any("range");
            let generator = signed_duration_in_range(range);
            let value = src.any_of("value", &generator);
            assert!(range.contains(&value));
            prop_smoke(src, "", &generator);
        });
    }

    #[test]
    fn signed_duration_gen_example() {
        check(|src| {
            let example: SignedDuration = src.any("example");
            let value = src.as_raw().any("value", Some(&example));
            assert_eq!(value, example);
        });
    }

    #[test]
    fn signed_duration_examples() {
        print_debug_examples(SignedDuration::arbitrary(), None, Ord::cmp);
    }

    #[test]
    fn timestamp_smoke() {
        check(|src| {
            prop_smoke(src, "Timestamp", Timestamp::arbitrary());
        });
    }

    #[test]
    fn timestamp_gen_in_range() {
        check(|src| {
            let range: Range<Timestamp> = src.any("range");
            let generator = timestamp_in_range(range);
            let value = src.any_of("value", &generator);
            assert!(range.contains(&value));
            prop_smoke(src, "", &generator);
        });
    }

    #[test]
    fn timestamp_gen_example() {
        check(|src| {
            let example: Timestamp = src.any("example");
            let value = src.as_raw().any("value", Some(&example));
            assert_eq!(value, example);
        });
    }

    #[test]
    fn timestamp_examples() {
        print_debug_examples(Timestamp::arbitrary(), None, Ord::cmp);
    }

    #[test]
    fn signed_duration_std_duration_tape_compatible() {
        check(|src| {
            let domain_max = Duration::new(i64::MAX as u64, MAX_SUBSEC_NANOS as u32);
            let a = src.any_of("a", make::duration_in_range(..=domain_max));
            let b = src.any_of("b", make::duration_in_range(..=domain_max));
            let (std_min, std_max) = (a.min(b), a.max(b));
            let to_jiff = |duration: Duration| {
                SignedDuration::new(duration.as_secs() as i64, duration.subsec_nanos() as i32)
            };
            let (jiff_min, jiff_max) = (to_jiff(std_min), to_jiff(std_max));
            let seed = src.any("seed");

            let std_tape = tape_for(seed, |src| {
                let _ = src.any_of("value", make::duration_in_range(std_min..=std_max));
            });
            let jiff_tape = tape_for(seed, |src| {
                let _ = src.any_of("value", signed_duration_in_range(jiff_min..=jiff_max));
            });
            assert_eq!(std_tape, jiff_tape);
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn timestamp_system_time_tape_compatible() {
        check(|src| {
            let a: std::time::SystemTime = src.any("a");
            let b: std::time::SystemTime = src.any("b");
            let (system_time_min, system_time_max) = (a.min(b), a.max(b));
            let to_jiff = |time: std::time::SystemTime| {
                let duration = time.duration_since(UNIX_EPOCH).unwrap();
                Timestamp::new(duration.as_secs() as i64, duration.subsec_nanos() as i32).unwrap()
            };
            let (timestamp_min, timestamp_max) =
                (to_jiff(system_time_min), to_jiff(system_time_max));
            let seed = src.any("seed");

            let std_tape = tape_for(seed, |src| {
                let _ = src.any_of(
                    "value",
                    make::system_time_in_range(system_time_min..=system_time_max),
                );
            });
            let jiff_tape = tape_for(seed, |src| {
                let _ = src.any_of("value", timestamp_in_range(timestamp_min..=timestamp_max));
            });
            assert_eq!(std_tape, jiff_tape);
        });
    }

    fn tape_for(seed: u32, prop: impl Fn(&mut Source)) -> Tape {
        Env::produce_tape(seed, TEMPERATURE_DEFAULT, BUDGET_DEFAULT, prop)
            .expect("generator produced an empty tape")
    }
}
