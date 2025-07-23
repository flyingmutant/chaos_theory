// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub(crate) struct IdentityHasher {
    value: u64,
}

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.value
    }

    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("internal error: using IdentityHasher for byte input")
    }

    fn write_u32(&mut self, u: u32) {
        self.value = u64::from(u);
    }

    #[inline]
    fn write_u64(&mut self, u: u64) {
        self.value = u;
    }

    fn write_i32(&mut self, i: i32) {
        self.value = i as u64;
    }

    fn write_i64(&mut self, i: i64) {
        self.value = i as u64;
    }
}

pub(crate) type IdentityBuildHasher = BuildHasherDefault<IdentityHasher>;

pub(crate) type NoHashSet<K> = std::collections::HashSet<K, IdentityBuildHasher>;
pub(crate) type NoHashMap<K, V> = std::collections::HashMap<K, V, IdentityBuildHasher>;
