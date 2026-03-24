// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{Arbitrary, Generator, MaybeOwned, make};
use uuid::{Uuid, Variant};

const UUID_V4_MASK: u128 = 0xFFFF_FFFF_FFFF_4FFF_BFFF_FFFF_FFFF_FFFF;
const UUID_V4_BITS: u128 = 0x0000_0000_0000_4000_8000_0000_0000_0000;

#[cfg_attr(docsrs, doc(cfg(feature = "uuid")))]
impl Arbitrary for Uuid {
    fn arbitrary() -> impl Generator<Item = Self> {
        uuid_v4()
    }
}

fn uuid_v4_from_u128(v: u128) -> Uuid {
    Uuid::from_u128(v & UUID_V4_MASK | UUID_V4_BITS)
}

/// Create a generator of version 4 [`Uuid`] values.
#[cfg_attr(docsrs, doc(cfg(feature = "uuid")))]
pub fn uuid_v4() -> impl Generator<Item = Uuid> {
    make::token().map_reversible(uuid_v4_from_u128, |uuid| {
        (uuid.get_version_num() == 4 && uuid.get_variant() == Variant::RFC4122)
            .then_some(MaybeOwned::Owned(uuid.as_u128()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, tests::prop_smoke};

    #[test]
    fn uuid_smoke() {
        check(|src| {
            prop_smoke(src, "uuid_v4", uuid_v4());
        });
    }

    #[test]
    fn uuid_arbitrary_is_v4() {
        check(|src| {
            let uuid: Uuid = src.any("uuid");
            assert_eq!(uuid.get_version_num(), 4);
            assert_eq!(uuid.get_variant(), Variant::RFC4122);
        });
    }
}
