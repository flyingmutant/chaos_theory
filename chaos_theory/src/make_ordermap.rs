// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{Arbitrary, Generator, range::SizeRange};
use crate::{Effect, OptionExt as _, SourceEx, generator::UNABLE_GENERATE_UNIQUE};
use core::fmt::Debug;
use core::hash::{BuildHasher, Hash};
use core::marker::PhantomData;
use core::ops::RangeBounds;
use ordermap::{OrderMap, OrderSet};

#[cfg_attr(docsrs, doc(cfg(feature = "ordermap")))]
impl<T, S> Arbitrary for OrderSet<T, S>
where
    T: Arbitrary + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        order_set(T::arbitrary())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ordermap")))]
impl<K, V, S> Arbitrary for OrderMap<K, V, S>
where
    K: Arbitrary + Hash + Eq,
    V: Arbitrary,
    S: BuildHasher + Default + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        order_map(K::arbitrary(), V::arbitrary())
    }
}

#[derive(Debug)]
struct OrderSet_<G, S> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<S>,
}

impl<G: Generator, S> Generator for OrderSet_<G, S>
where
    G::Item: Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    type Item = OrderSet<G::Item, S>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<orderset>",
            example_seq,
            self.size,
            |n| OrderSet::with_capacity_and_hasher(n, S::default()),
            |v, src, example| {
                let elem = self.elem.next(src, example);
                let ok = v.insert(elem);
                if ok { Effect::Success } else { Effect::Noop }
            },
        );
        res.assume_some_msg(UNABLE_GENERATE_UNIQUE)
    }
}

/// Create a [`OrderSet`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "ordermap")))]
pub fn order_set<T, S>(elem: impl Generator<Item = T>) -> impl Generator<Item = OrderSet<T, S>>
where
    T: Debug + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    order_set_with_size::<T, S>(elem, ..)
}

/// Create a [`OrderSet`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "ordermap")))]
pub fn order_set_with_size<T, S>(
    elem: impl Generator<Item = T>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = OrderSet<T, S>>
where
    T: Debug + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    let size = SizeRange::new(size);
    OrderSet_ {
        elem,
        size,
        _marker: PhantomData,
    }
}

#[derive(Debug)]
struct OrderMap_<GK, GV, S> {
    key: GK,
    value: GV,
    size: SizeRange,
    _marker: PhantomData<S>,
}

impl<GK: Generator, GV: Generator, S> Generator for OrderMap_<GK, GV, S>
where
    GK::Item: Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    type Item = OrderMap<GK::Item, GV::Item, S>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<ordermap>",
            example_seq,
            self.size,
            |n| OrderMap::with_capacity_and_hasher(n, S::default()),
            |v, src, example| {
                use ordermap::map::Entry::{Occupied, Vacant};
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

/// Create a [`OrderMap`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "ordermap")))]
pub fn order_map<K, V, S>(
    key: impl Generator<Item = K>,
    value: impl Generator<Item = V>,
) -> impl Generator<Item = OrderMap<K, V, S>>
where
    K: Debug + Hash + Eq,
    V: Debug,
    S: BuildHasher + Default + Debug,
{
    order_map_with_size::<K, V, S>(key, value, ..)
}

/// Create a [`OrderMap`] generator with a specified size range.
#[cfg_attr(docsrs, doc(cfg(feature = "ordermap")))]
pub fn order_map_with_size<K, V, S>(
    key: impl Generator<Item = K>,
    value: impl Generator<Item = V>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = OrderMap<K, V, S>>
where
    K: Debug + Hash + Eq,
    V: Debug,
    S: BuildHasher + Default + Debug,
{
    let size = SizeRange::new(size);
    OrderMap_ {
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
    fn ordermap_smoke() {
        const MAX_SIZE: usize = 5; // low limit to not run out of the tape
        check(|src| {
            prop_smoke(
                src,
                "OrderSet",
                make::ordermap::order_set_with_size::<_, RandomState>(
                    make::arbitrary::<i32>(),
                    ..MAX_SIZE,
                ),
            );
            prop_smoke(
                src,
                "OrderMap",
                make::ordermap::order_map_with_size::<_, _, RandomState>(
                    make::arbitrary::<i32>(),
                    make::arbitrary::<i32>(),
                    ..MAX_SIZE,
                ),
            );
        });
    }
}
