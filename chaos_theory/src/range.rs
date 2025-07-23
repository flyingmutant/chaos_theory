// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    fmt::{Debug, Display},
    ops::{Bound, RangeBounds},
};

use crate::{Arbitrary, Generator, Ranged, make};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Range<R> {
    pub(crate) min: R,
    pub(crate) max: R,
}

pub(crate) type SizeRange = Range<usize>;

impl<R: Ranged> Range<R> {
    #[track_caller]
    pub(crate) fn new(r: impl RangeBounds<R>) -> Self {
        let min = match r.start_bound() {
            Bound::Unbounded => R::MIN,
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.next_up().expect("invalid start bound"),
        };
        let max = match r.end_bound() {
            Bound::Unbounded => R::MAX,
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.next_down().expect("invalid end bound"),
        };
        assert!(min <= max, "invalid range {min:?}..={max:?}");
        Self::new_raw(min, max)
    }

    #[track_caller]
    pub(crate) fn new_raw(min: R, max: R) -> Self {
        debug_assert!(min <= max, "invalid range {min:?}..={max:?}");
        Self { min, max }
    }

    pub(crate) fn zero_split(&self) -> (Option<Self>, Option<Self>) {
        if self.min >= R::ZERO {
            (None, Some(*self))
        } else if self.max < R::ZERO {
            (Some(*self), None)
        } else {
            (
                Some(Self {
                    min: self.min,
                    max: R::next_down(R::ZERO)
                        .expect("internal error: unable to next_down() for ZERO"),
                }),
                Some(Self {
                    min: R::ZERO,
                    max: self.max,
                }),
            )
        }
    }
}

impl<R: Ranged + Arbitrary> Arbitrary for Range<R> {
    fn arbitrary() -> impl Generator<Item = Self> {
        make::from_fn::<Self>(|src, example| {
            let mut min = src.any("min?", example.map(|r| &r.min));
            let mut max = src.any("max?", example.map(|r| &r.max));
            if min > max {
                (min, max) = (max, min);
            }
            Self::new_raw(min, max)
        })
    }
}

impl<R: Ranged> RangeBounds<R> for Range<R> {
    fn start_bound(&self) -> Bound<&R> {
        if self.min == R::MIN {
            Bound::Unbounded
        } else {
            Bound::Included(&self.min)
        }
    }

    fn end_bound(&self) -> Bound<&R> {
        if self.max == R::MAX {
            Bound::Unbounded
        } else {
            Bound::Included(&self.max)
        }
    }

    fn contains<U>(&self, item: &U) -> bool
    where
        R: PartialOrd<U>,
        U: ?Sized + PartialOrd<R>,
    {
        *item >= self.min && *item <= self.max
    }
}

impl<R: Ranged> Debug for Range<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (start, end) = (self.min, self.max);
        if start == R::MIN && end == R::MAX {
            write!(f, "..")
        } else if start == R::MIN {
            write!(f, "..={end:?}")
        } else if end == R::MAX {
            write!(f, "{start:?}..")
        } else {
            write!(f, "{start:?}..={end:?}")
        }
    }
}

impl<R: Ranged> Display for Range<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        <Self as Debug>::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::Range;
    use crate::{check, make, tests::prop_smoke};

    #[test]
    fn range_smoke() {
        check(|src| {
            prop_smoke(src, "Range", make::arbitrary::<Range<i32>>());
        });
    }
}
