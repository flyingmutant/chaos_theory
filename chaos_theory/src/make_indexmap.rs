// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{Arbitrary, Generator, range::SizeRange};
use crate::{Effect, OptionExt as _, SourceRaw, UNABLE_GENERATE_UNIQUE};
use core::fmt::Debug;
use core::hash::{BuildHasher, Hash};
use core::marker::PhantomData;
use core::ops::RangeBounds;
use indexmap::{IndexMap, IndexSet};

impl<T, S> Arbitrary for IndexSet<T, S>
where
    T: Arbitrary + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        index_set(T::arbitrary())
    }
}

impl<K, V, S> Arbitrary for IndexMap<K, V, S>
where
    K: Arbitrary + Hash + Eq,
    V: Arbitrary,
    S: BuildHasher + Default + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        index_map(K::arbitrary(), V::arbitrary())
    }
}

#[derive(Debug)]
struct IndexSet_<G, S> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<S>,
}

impl<G: Generator, S> Generator for IndexSet_<G, S>
where
    G::Item: Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    type Item = IndexSet<G::Item, S>;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<indexset>",
            example_seq,
            self.size,
            |n| IndexSet::with_capacity_and_hasher(n, S::default()),
            |v, src, example| {
                let elem = self.elem.next(src, example);
                let ok = v.insert(elem);
                if ok { Effect::Success } else { Effect::Noop }
            },
        );
        res.assume_some_msg(UNABLE_GENERATE_UNIQUE)
    }
}

/// Create a [`IndexSet`] generator.
pub fn index_set<T, S>(elem: impl Generator<Item = T>) -> impl Generator<Item = IndexSet<T, S>>
where
    T: Debug + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    index_set_with_size::<T, S>(elem, ..)
}

/// Create a [`IndexSet`] generator with a specified size range.
pub fn index_set_with_size<T, S>(
    elem: impl Generator<Item = T>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = IndexSet<T, S>>
where
    T: Debug + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    let size = SizeRange::new(size);
    IndexSet_ {
        elem,
        size,
        _marker: PhantomData,
    }
}

#[derive(Debug)]
struct IndexMap_<GK, GV, S> {
    key: GK,
    value: GV,
    size: SizeRange,
    _marker: PhantomData<S>,
}

impl<GK: Generator, GV: Generator, S> Generator for IndexMap_<GK, GV, S>
where
    GK::Item: Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    type Item = IndexMap<GK::Item, GV::Item, S>;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<indexmap>",
            example_seq,
            self.size,
            |n| IndexMap::with_capacity_and_hasher(n, S::default()),
            |v, src, example| {
                use indexmap::map::Entry::{Occupied, Vacant};
                let key = src.any_of("<key>", &self.key, example.map(|e| e.0));
                match v.entry(key) {
                    Occupied(_) => Effect::Noop,
                    Vacant(e) => {
                        let val = src.any_of("<value>", &self.value, example.map(|e| e.1));
                        e.insert(val);
                        Effect::Success
                    }
                }
            },
        );
        res.assume_some_msg(UNABLE_GENERATE_UNIQUE)
    }
}

/// Create a [`IndexMap`] generator.
pub fn index_map<K, V, S>(
    key: impl Generator<Item = K>,
    value: impl Generator<Item = V>,
) -> impl Generator<Item = IndexMap<K, V, S>>
where
    K: Debug + Hash + Eq,
    V: Debug,
    S: BuildHasher + Default + Debug,
{
    index_map_with_size::<K, V, S>(key, value, ..)
}

/// Create a [`IndexMap`] generator with a specified size range.
pub fn index_map_with_size<K, V, S>(
    key: impl Generator<Item = K>,
    value: impl Generator<Item = V>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = IndexMap<K, V, S>>
where
    K: Debug + Hash + Eq,
    V: Debug,
    S: BuildHasher + Default + Debug,
{
    let size = SizeRange::new(size);
    IndexMap_ {
        key,
        value,
        size,
        _marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use crate::{check, make, tests::prop_smoke};
    use std::hash::RandomState;

    #[test]
    fn indexmap_smoke() {
        const MAX_SIZE: usize = 5; // low limit to not run out of the tape
        check(|src| {
            prop_smoke(
                src,
                "IndexSet",
                make::indexmap::index_set_with_size::<_, RandomState>(
                    make::arbitrary::<i32>(),
                    ..MAX_SIZE,
                ),
            );
            prop_smoke(
                src,
                "IndexMap",
                make::indexmap::index_map_with_size::<_, _, RandomState>(
                    make::arbitrary::<i32>(),
                    make::arbitrary::<i32>(),
                    ..MAX_SIZE,
                ),
            );
        });
    }
}
