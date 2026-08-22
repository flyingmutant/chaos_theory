// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#[cfg(test)]
#[path = "tests_std.rs"]
mod std_impl;

#[cfg(test)]
pub(crate) use std_impl::*;

#[cfg(all(not(feature = "std"), feature = "derive"))]
mod derive_nostd_smoke {
    use alloc::{string::String, vec::Vec};

    use crate::{Arbitrary, Env};

    #[derive(Clone, Debug, PartialEq, Eq, Arbitrary)]
    struct Sample {
        id: u32,
        name: String,
        data: Vec<u8>,
    }

    #[allow(dead_code)]
    fn smoke_via_env() {
        let mut env = Env::new();
        let _value: Sample = env.example(None);
    }

    #[allow(dead_code)]
    fn smoke_via_source_raw() {
        let mut env = Env::custom().env(false);
        let mut src = env.__start_from_nothing(true);
        let _value: Sample = src.as_ex().any("sample", None);
    }
}
