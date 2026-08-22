// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{Arbitrary, Generator, make_collection::value_fingerprint, range::SizeRange};
use crate::{Effect, OptionExt as _, SourceEx, generator::UNABLE_GENERATE_UNIQUE};
use alloc::vec::Vec;
use core::fmt::Debug;
use core::hash::{BuildHasher, Hash};
use core::marker::PhantomData;
use core::ops::RangeBounds;
use hashbrown::{HashMap, HashSet};

#[cfg_attr(docsrs, doc(cfg(feature = "hashbrown")))]
impl<T, S> Arbitrary for HashSet<T, S>
where
    T: Arbitrary + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        hash_set(T::arbitrary())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "hashbrown")))]
impl<K, V, S> Arbitrary for HashMap<K, V, S>
where
    K: Arbitrary + Hash + Eq,
    V: Arbitrary,
    S: BuildHasher + Default + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        hash_map(K::arbitrary(), V::arbitrary())
    }
}

#[derive(Debug)]
struct HashSet_<G, S> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<S>,
}

impl<G: Generator, S> Generator for HashSet_<G, S>
where
    G::Item: Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    type Item = HashSet<G::Item, S>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| {
            let mut items: Vec<_> = e.iter().collect();
            items.sort_unstable_by_key(|value| value_fingerprint(*value));
            items.into_iter()
        });
        let res = src.repeat(
            "<hashset>",
            example_seq,
            self.size,
            |n| HashSet::with_capacity_and_hasher(n, S::default()),
            |v, src, example| {
                let elem = self.elem.next(src, example);
                let ok = v.insert(elem);
                if ok { Effect::Success } else { Effect::Noop }
            },
        );
        res.assume_some_msg(UNABLE_GENERATE_UNIQUE)
    }
}

/// Create a [`HashSet`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "hashbrown")))]
pub fn hash_set<T, S>(elem: impl Generator<Item = T>) -> impl Generator<Item = HashSet<T, S>>
where
    T: Debug + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    hash_set_with_size::<T, S>(elem, ..)
}

/// Create a [`HashSet`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "hashbrown")))]
pub fn hash_set_with_size<T, S>(
    elem: impl Generator<Item = T>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = HashSet<T, S>>
where
    T: Debug + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    let size = SizeRange::new(size);
    HashSet_ {
        elem,
        size,
        _marker: PhantomData,
    }
}

#[derive(Debug)]
struct HashMap_<GK, GV, S> {
    key: GK,
    value: GV,
    size: SizeRange,
    _marker: PhantomData<S>,
}

impl<GK: Generator, GV: Generator, S> Generator for HashMap_<GK, GV, S>
where
    GK::Item: Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    type Item = HashMap<GK::Item, GV::Item, S>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| {
            let mut items: Vec<_> = e.iter().collect();
            items.sort_unstable_by_key(|(key, _)| value_fingerprint(*key));
            items.into_iter()
        });
        let res = src.repeat(
            "<hashmap>",
            example_seq,
            self.size,
            |n| HashMap::with_capacity_and_hasher(n, S::default()),
            |v, src, example| {
                use hashbrown::hash_map::Entry::{Occupied, Vacant};
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

/// Create a [`HashMap`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "hashbrown")))]
pub fn hash_map<K, V, S>(
    key: impl Generator<Item = K>,
    value: impl Generator<Item = V>,
) -> impl Generator<Item = HashMap<K, V, S>>
where
    K: Debug + Hash + Eq,
    V: Debug,
    S: BuildHasher + Default + Debug,
{
    hash_map_with_size::<K, V, S>(key, value, ..)
}

/// Create a [`HashMap`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "hashbrown")))]
pub fn hash_map_with_size<K, V, S>(
    key: impl Generator<Item = K>,
    value: impl Generator<Item = V>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = HashMap<K, V, S>>
where
    K: Debug + Hash + Eq,
    V: Debug,
    S: BuildHasher + Default + Debug,
{
    let size = SizeRange::new(size);
    HashMap_ {
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
    fn hashbrown_smoke() {
        const MAX_SIZE: usize = 5; // low limit to not run out of the tape
        check(|src| {
            prop_smoke(
                src,
                "HashSet",
                make::hashbrown::hash_set_with_size::<_, RandomState>(
                    make::arbitrary::<i32>(),
                    ..MAX_SIZE,
                ),
            );
            prop_smoke(
                src,
                "HashMap",
                make::hashbrown::hash_map_with_size::<_, _, RandomState>(
                    make::arbitrary::<i32>(),
                    make::arbitrary::<i32>(),
                    ..MAX_SIZE,
                ),
            );
        });
    }
}
