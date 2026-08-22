// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use bstr::BString;
use core::ops::RangeBounds;

use crate::{Arbitrary, Generator, make};

#[cfg_attr(docsrs, doc(cfg(feature = "bstr")))]
impl Arbitrary for BString {
    fn arbitrary() -> impl Generator<Item = Self> {
        bstring(make::arbitrary())
    }
}

/// Create a [`BString`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "bstr")))]
pub fn bstring(elem: impl Generator<Item = u8>) -> impl Generator<Item = BString> {
    bstring_n(elem, ..)
}

/// Create a [`BString`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "bstr")))]
pub fn bstring_n(
    elem: impl Generator<Item = u8>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = BString> {
    // TODO: rework generic string generators so that they can produce non-UTF-8 data and use this mode here.
    make::vec_n(elem, size).map_into_deref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, make, tests::prop_smoke};

    #[test]
    fn bstring_smoke() {
        check(|src| {
            prop_smoke(src, "BString", bstring_n(make::arbitrary(), ..));
        });
    }
}
