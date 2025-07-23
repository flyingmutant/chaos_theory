// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![expect(clippy::allow_attributes)]

use crate::{Arbitrary, Generator, SourceRaw};

macro_rules! define_tuple_arbitrary {
    ($($params: ident)*, $($ixs: tt)*) => {
        impl<$($params: Generator, )*> Generator for ($($params, )*) {
            type Item = ($($params::Item, )*);

            #[allow(unused_variables, clippy::unused_unit)]
            fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
                ($(src.any_of(stringify!($ixs), &self.$ixs, example.map(|e| &e.$ixs)), )*)
            }
        }

        impl<$($params: Arbitrary, )*> Arbitrary for ($($params, )*) {
            #[allow(clippy::unused_unit)]
            fn arbitrary() -> impl Generator<Item = Self> {
                ($($params::arbitrary(), )*)
            }
        }
    };
}

define_tuple_arbitrary!(,);
define_tuple_arbitrary!(A, 0);
define_tuple_arbitrary!(A B, 0 1);
define_tuple_arbitrary!(A B C, 0 1 2);
define_tuple_arbitrary!(A B C D, 0 1 2 3);
define_tuple_arbitrary!(A B C D E, 0 1 2 3 4);
define_tuple_arbitrary!(A B C D E F, 0 1 2 3 4 5);
define_tuple_arbitrary!(A B C D E F G, 0 1 2 3 4 5 6);
define_tuple_arbitrary!(A B C D E F G H, 0 1 2 3 4 5 6 7);
define_tuple_arbitrary!(A B C D E F G H I, 0 1 2 3 4 5 6 7 8);
define_tuple_arbitrary!(A B C D E F G H I J, 0 1 2 3 4 5 6 7 8 9);
define_tuple_arbitrary!(A B C D E F G H I J K, 0 1 2 3 4 5 6 7 8 9 10);
define_tuple_arbitrary!(A B C D E F G H I J K L, 0 1 2 3 4 5 6 7 8 9 10 11);

#[cfg(test)]
mod tests {
    use crate::{check, make, tests::prop_smoke};

    #[test]
    fn tuple_smoke() {
        check(|src| {
            prop_smoke(src, "(i32, i32, i32)", make::arbitrary::<(i32, i32, i32)>());
        });
    }
}
