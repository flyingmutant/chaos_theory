// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    fmt::Debug,
    ops::{Bound, RangeBounds},
    time::Duration,
};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{Arbitrary, Generator, Ranged, SourceRaw, Tweak, make, range::Range};

const NANOS_PER_SEC: u32 = 1_000_000_000;
const DURATION_SMALLEST: Duration = Duration::new(0, 1);
const DURATION_SMALLEST_SYSTEM_TIME: Duration = Duration::new(0, 100); // match Windows clock precision
const DURATION_MAX_FROM_EPOCH: Duration = Duration::from_secs(500 * 365 * 24 * 60 * 60); // 2470/01/01 00:00:00; fits in 64 bits as nanoseconds
const DURATION_ANCHOR_FROM_EPOCH: Duration = Duration::from_secs(946684800); // 2000/01/01 00:00:00

impl Arbitrary for Duration {
    fn arbitrary() -> impl Generator<Item = Self> {
        duration_in_range(..)
    }
}

impl Arbitrary for SystemTime {
    fn arbitrary() -> impl Generator<Item = Self> {
        system_time_in_range(..)
    }
}

impl Generator for Duration_ {
    type Item = Duration;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let secs = src.any_of(
            "<secs>",
            make::int_in_range(self.secs),
            example.map(Duration::as_secs).as_ref(),
        );
        let (nanos_min, nanos_max) = if secs == self.secs.min && secs == self.secs.max {
            (self.nanos_start, self.nanos_end)
        } else if secs == self.secs.min {
            (self.nanos_start, NANOS_PER_SEC - 1)
        } else if secs == self.secs.max {
            (0, self.nanos_end)
        } else {
            (0, NANOS_PER_SEC - 1)
        };
        let nanos = src.any_of(
            "<nanos>",
            make::int_in_range(nanos_min..=nanos_max),
            example.map(Duration::subsec_nanos).as_ref(),
        );
        Duration::new(secs, nanos)
    }
}

struct Duration_ {
    range: Range<Duration>,
    secs: Range<u64>,
    nanos_start: u32,
    nanos_end: u32,
}

impl Debug for Duration_ {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Duration").field(&self.range).finish()
    }
}

impl Ranged for Duration {
    const ZERO: Self = Self::ZERO;
    const MIN: Self = Self::ZERO;
    const MAX: Self = Self::MAX;

    fn next_up(self) -> Option<Self> {
        self.checked_add(DURATION_SMALLEST)
    }

    fn next_down(self) -> Option<Self> {
        if self.is_zero() {
            Some(Self::ZERO)
        } else {
            self.checked_sub(DURATION_SMALLEST)
        }
    }
}

/// Create a generator of [`Duration`] in range.
pub fn duration_in_range(r: impl RangeBounds<Duration>) -> impl Generator<Item = Duration> {
    let range = Range::new(r);
    Duration_ {
        range,
        secs: Range::new_raw(range.min.as_secs(), range.max.as_secs()),
        nanos_start: range.min.subsec_nanos(),
        nanos_end: range.max.subsec_nanos(),
    }
}

fn duration_since_epoch(t: SystemTime) -> Duration {
    t.duration_since(UNIX_EPOCH)
        .expect("internal error: unable to get duration since epoch")
}

fn time_since_epoch(d: Duration) -> SystemTime {
    UNIX_EPOCH
        .checked_add(d)
        .expect("internal error: unable to get time since epoch")
}

impl Generator for SystemTime_ {
    type Item = SystemTime;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example = example.map(|t| duration_since_epoch(*t));
        let example_before = example.map(|d| d < DURATION_ANCHOR_FROM_EPOCH);
        let (example_before, forced) = match (self.before, self.after) {
            (Some(_), Some(_)) => (example_before, false),
            (Some(_), None) => (Some(true), true),
            (None, Some(_)) => (Some(false), true),
            (None, None) => unreachable!(),
        };
        if forced {
            src.mark_next_choice_forced();
        }
        let ix =
            src.as_mut()
                .choose_index(2, example_before.map(usize::from), Tweak::SystemTimeEpoch);
        let r = if ix == 0 { &self.after } else { &self.before };
        let r = r.expect("internal error: range not set");
        let d = duration_in_range(r).next(src, example.as_ref());
        time_since_epoch(d)
    }
}

struct SystemTime_ {
    range: Range<Duration>,
    before: Option<Range<Duration>>,
    after: Option<Range<Duration>>,
}

impl Debug for SystemTime_ {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SystemTime")
            .field(&time_since_epoch(self.range.min))
            .field(&time_since_epoch(self.range.max))
            .finish()
    }
}

/// Create a generator of [`SystemTime`] in range.
///
/// Left range bound defaults to [`UNIX_EPOCH`], right range bound defaults to 500 years since [`UNIX_EPOCH`].
#[expect(clippy::missing_panics_doc)]
pub fn system_time_in_range(r: impl RangeBounds<SystemTime>) -> impl Generator<Item = SystemTime> {
    let min = match r.start_bound() {
        Bound::Unbounded => UNIX_EPOCH,
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n
            .checked_add(DURATION_SMALLEST_SYSTEM_TIME)
            .expect("invalid start bound"),
    };
    let max = match r.end_bound() {
        Bound::Unbounded => time_since_epoch(DURATION_MAX_FROM_EPOCH),
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n
            .checked_sub(DURATION_SMALLEST_SYSTEM_TIME)
            .expect("invalid end bound"),
    };
    assert!(min <= max, "invalid range {min:?}..={max:?}");
    let range = Range::new_raw(duration_since_epoch(min), duration_since_epoch(max));
    let (before, after) = if range.min >= DURATION_ANCHOR_FROM_EPOCH {
        (None, Some(range))
    } else if range.max < DURATION_ANCHOR_FROM_EPOCH {
        (Some(range), None)
    } else {
        (
            // We include anchor in both ranges to avoid DURATION_SMALLEST_SYSTEM_TIME hole between them.
            Some(Range::new_raw(range.min, DURATION_ANCHOR_FROM_EPOCH)),
            Some(Range::new_raw(DURATION_ANCHOR_FROM_EPOCH, range.max)),
        )
    };
    SystemTime_ {
        range,
        before,
        after,
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Arbitrary as _, check, make,
        range::Range,
        tests::{print_debug_examples, prop_smoke},
    };
    use core::{ops::RangeBounds as _, time::Duration};
    use std::time::SystemTime;

    #[test]
    fn duration_smoke() {
        check(|src| {
            prop_smoke(src, "Duration", Duration::arbitrary());
        });
    }

    #[test]
    fn duration_gen_in_range() {
        check(|src| {
            let r: Range<Duration> = src.any("r");
            let g = make::duration_in_range(r);
            let value = src.any_of("value", &g);
            assert!(r.contains(&value));
            prop_smoke(src, "", &g);
        });
    }

    #[test]
    fn duration_gen_example() {
        check(|src| {
            let example: Duration = src.any("example");
            let d = src.as_raw().any("d", Some(&example));
            assert_eq!(d, example);
        });
    }

    #[test]
    fn duration_examples() {
        print_debug_examples(&make::arbitrary::<Duration>(), None, Ord::cmp);
    }

    #[test]
    fn system_time_smoke() {
        check(|src| {
            prop_smoke(src, "SystemTime", SystemTime::arbitrary());
        });
    }

    #[test]
    fn system_time_gen_in_range() {
        check(|src| {
            let (a, b): (SystemTime, SystemTime) = src.any("a, b");
            let r = a.min(b)..=a.max(b);
            let g = make::system_time_in_range(r.clone());
            let value = src.any_of("value", &g);
            assert!(r.contains(&value));
            prop_smoke(src, "", &g);
        });
    }

    #[test]
    fn system_time_gen_example() {
        check(|src| {
            let example: SystemTime = src.any("example");
            let d = src.as_raw().any("d", Some(&example));
            assert_eq!(d, example);
        });
    }

    #[test]
    fn system_time_examples() {
        print_debug_examples(&make::arbitrary::<SystemTime>(), None, Ord::cmp);
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use crate::{Arbitrary as _, tests::bench_gen_next};
    use core::time::Duration;
    use std::time::SystemTime;

    #[bench]
    fn gen_duration(b: &mut test::Bencher) {
        let g = Duration::arbitrary();
        bench_gen_next(b, &g);
    }

    #[bench]
    fn gen_system_time(b: &mut test::Bencher) {
        let g = SystemTime::arbitrary();
        bench_gen_next(b, &g);
    }
}
