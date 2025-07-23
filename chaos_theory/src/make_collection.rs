// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, BinaryHeap, LinkedList, VecDeque},
    rc::Rc,
    sync::Arc,
};
use core::{
    fmt::Debug,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    ops::{Deref, RangeBounds},
};
use std::collections::{HashMap, HashSet};

use crate::{
    Arbitrary, Effect, Generator, OptionExt as _, SourceRaw, UNABLE_GENERATE_UNIQUE,
    range::SizeRange,
};

impl<T: Arbitrary> Arbitrary for Box<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        T::arbitrary().map_into_deref()
    }
}

impl<T: Arbitrary> Arbitrary for Rc<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        T::arbitrary().map_into_deref()
    }
}

impl<T: Arbitrary> Arbitrary for Arc<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        T::arbitrary().map_into_deref()
    }
}

impl<T: Arbitrary, const N: usize> Arbitrary for [T; N] {
    fn arbitrary() -> impl Generator<Item = Self> {
        array(T::arbitrary())
    }
}

impl<T: Arbitrary> Arbitrary for Vec<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        vec(T::arbitrary())
    }
}

impl<T: Arbitrary> Arbitrary for Box<[T]> {
    fn arbitrary() -> impl Generator<Item = Self> {
        slice(T::arbitrary())
    }
}

impl<T: Arbitrary> Arbitrary for Rc<[T]> {
    fn arbitrary() -> impl Generator<Item = Self> {
        slice(T::arbitrary())
    }
}

impl<T: Arbitrary> Arbitrary for Arc<[T]> {
    fn arbitrary() -> impl Generator<Item = Self> {
        slice(T::arbitrary())
    }
}

impl<T: Arbitrary + Clone> Arbitrary for Cow<'_, [T]> {
    fn arbitrary() -> impl Generator<Item = Self> {
        slice(T::arbitrary())
    }
}

impl<T: Arbitrary> Arbitrary for VecDeque<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        T::arbitrary().collect::<Self>()
    }
}

impl<T: Arbitrary> Arbitrary for LinkedList<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        T::arbitrary().collect::<Self>()
    }
}

impl<T: Arbitrary + Ord> Arbitrary for BinaryHeap<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        T::arbitrary().collect::<Self>()
    }
}

impl<T: Arbitrary + Ord> Arbitrary for BTreeSet<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        btree_set(T::arbitrary())
    }
}

impl<K: Arbitrary + Ord, V: Arbitrary> Arbitrary for BTreeMap<K, V> {
    fn arbitrary() -> impl Generator<Item = Self> {
        btree_map(K::arbitrary(), V::arbitrary())
    }
}

impl<T, S> Arbitrary for HashSet<T, S>
where
    T: Arbitrary + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    fn arbitrary() -> impl Generator<Item = Self> {
        hash_set(T::arbitrary())
    }
}

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
struct Array<G, const N: usize> {
    elem: G,
}

impl<G: Generator, const N: usize> Generator for Array<G, N> {
    type Item = [G::Item; N];

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        // Avoid `MaybeUninit` because it requires `unsafe`.
        let mut i = 0;
        [(); N].map(|()| {
            let example = example.map(|e| &e[i]);
            i += 1;
            src.any_of("<array-elem>", &self.elem, example)
        })
    }
}

/// Create an array generator.
pub fn array<T, const N: usize>(elem: impl Generator<Item = T>) -> impl Generator<Item = [T; N]>
where
    T: Debug,
{
    Array { elem }
}

// This exists mainly because `Extend::extend_reserve` is not stable,
// (meaning `Generator::collect_n` is sub-optimal) and `Vec` is the most common
// and performance-critical collection.
#[derive(Debug)]
struct Vec_<G, T> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<T>,
}

impl<G, T> Generator for Vec_<G, T>
where
    G: Generator,
    T: From<Vec<G::Item>> + Deref<Target = [G::Item]> + Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let mut v = Vec::new();
        next_vec_impl(
            src,
            example.map(Deref::deref),
            &mut v,
            &self.elem,
            self.size,
        );
        v.into()
    }
}

pub(crate) fn next_vec_impl<G: Generator>(
    src: &mut SourceRaw,
    example: Option<&[G::Item]>,
    v: &mut Vec<G::Item>,
    elem: G,
    size: SizeRange,
) {
    let example_seq = example.map(IntoIterator::into_iter);
    let res = src.repeat(
        "<vec>",
        example_seq,
        size,
        |n| {
            v.reserve(n);
            v
        },
        |v, src, example| {
            let elem = elem.next(src, example);
            v.push(elem);
            Effect::Success
        },
    );
    res.expect("internal error: vec element repeat should not fail");
}

/// Create a [`Vec`] generator.
pub fn vec<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = Vec<T>>
where
    T: Debug,
{
    vec_with_size::<T>(elem, ..)
}

/// Create a [`Vec`] generator with a specified size range.
pub fn vec_with_size<T>(
    elem: impl Generator<Item = T>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = Vec<T>>
where
    T: Debug,
{
    slice_with_size(elem, size)
}

/// Create an owned slice generator.
///
/// Examples of standard owned slices:
///
/// - [`Box<[T]>`](Box)
/// - [`Rc<[T]>`](alloc::rc::Rc)
/// - [`Arc<[T]>`](alloc::sync::Arc)
/// - [`Cow<'_, [T]>`](alloc::borrow::Cow)
pub fn slice<S, T>(elem: impl Generator<Item = T>) -> impl Generator<Item = S>
where
    S: From<Vec<T>> + Deref<Target = [T]> + Debug,
    T: Debug,
{
    slice_with_size(elem, ..)
}

/// Create an owned slice generator with a specified size range.
///
/// Examples of standard owned slices:
///
/// - [`Box<[T]>`](Box)
/// - [`Rc<[T]>`](alloc::rc::Rc)
/// - [`Arc<[T]>`](alloc::sync::Arc)
/// - [`Cow<'_, [T]>`](alloc::borrow::Cow)
pub fn slice_with_size<S, T>(
    elem: impl Generator<Item = T>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = S>
where
    S: From<Vec<T>> + Deref<Target = [T]> + Debug,
    T: Debug,
{
    let size = SizeRange::new(size);
    Vec_ {
        elem,
        size,
        _marker: PhantomData,
    }
}

#[derive(Debug)]
struct BTreeSet_<G> {
    elem: G,
    size: SizeRange,
}

impl<G: Generator> Generator for BTreeSet_<G>
where
    G::Item: Ord,
{
    type Item = BTreeSet<G::Item>;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<btreeset>",
            example_seq,
            self.size,
            |_n| BTreeSet::new(),
            |v, src, example| {
                let elem = self.elem.next(src, example);
                let ok = v.insert(elem);
                if ok { Effect::Success } else { Effect::Noop }
            },
        );
        res.assume_some_msg(UNABLE_GENERATE_UNIQUE)
    }
}

/// Create a [`BTreeSet`] generator.
pub fn btree_set<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = BTreeSet<T>>
where
    T: Debug + Ord,
{
    btree_set_with_size::<T>(elem, ..)
}

/// Create a [`BTreeSet`] generator with a specified size range.
pub fn btree_set_with_size<T>(
    elem: impl Generator<Item = T>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = BTreeSet<T>>
where
    T: Debug + Ord,
{
    let size = SizeRange::new(size);
    BTreeSet_ { elem, size }
}

#[derive(Debug)]
struct BTreeMap_<GK, GV> {
    key: GK,
    value: GV,
    size: SizeRange,
}

impl<GK: Generator, GV: Generator> Generator for BTreeMap_<GK, GV>
where
    GK::Item: Ord,
{
    type Item = BTreeMap<GK::Item, GV::Item>;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<btreemap>",
            example_seq,
            self.size,
            |_n| BTreeMap::new(),
            |v, src, example| {
                use alloc::collections::btree_map::Entry::{Occupied, Vacant};
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

/// Create a [`BTreeMap`] generator.
pub fn btree_map<K, V>(
    key: impl Generator<Item = K>,
    value: impl Generator<Item = V>,
) -> impl Generator<Item = BTreeMap<K, V>>
where
    K: Debug + Ord,
    V: Debug,
{
    btree_map_with_size::<K, V>(key, value, ..)
}

/// Create a [`BTreeMap`] generator with a specified size range.
pub fn btree_map_with_size<K, V>(
    key: impl Generator<Item = K>,
    value: impl Generator<Item = V>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = BTreeMap<K, V>>
where
    K: Debug + Ord,
    V: Debug,
{
    let size = SizeRange::new(size);
    BTreeMap_ { key, value, size }
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

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
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
pub fn hash_set<T, S>(elem: impl Generator<Item = T>) -> impl Generator<Item = HashSet<T, S>>
where
    T: Debug + Hash + Eq,
    S: BuildHasher + Default + Debug,
{
    hash_set_with_size::<T, S>(elem, ..)
}

/// Create a [`HashSet`] generator with a specified size range.
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

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<hashmap>",
            example_seq,
            self.size,
            |n| HashMap::with_capacity_and_hasher(n, S::default()),
            |v, src, example| {
                use std::collections::hash_map::Entry::{Occupied, Vacant};
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
    use super::*;
    use crate::{
        check, make,
        range::Range,
        tests::{print_debug_examples, prop_smoke},
    };
    use std::hash::RandomState;

    #[test]
    fn vec_u64_examples() {
        let g = u64::arbitrary().collect::<Vec<_>>();
        print_debug_examples(g, None, Ord::cmp);
    }

    #[test]
    fn collection_smoke() {
        const MAX_SIZE: usize = 5; // low limit to not run out of the tape
        check(|src| {
            prop_smoke(src, "Array", make::arbitrary::<[i32; MAX_SIZE]>());
            prop_smoke(
                src,
                "Vec",
                make::vec_with_size(make::arbitrary::<i32>(), ..MAX_SIZE),
            );
            prop_smoke(
                src,
                "VecDeque",
                make::arbitrary::<i32>().collect_n::<VecDeque<_>>(..MAX_SIZE),
            );
            prop_smoke(
                src,
                "LinkedList",
                make::arbitrary::<i32>().collect_n::<LinkedList<_>>(..MAX_SIZE),
            );
            prop_smoke(
                src,
                "BTreeSet",
                make::btree_set_with_size(make::arbitrary::<i32>(), ..MAX_SIZE),
            );
            prop_smoke(
                src,
                "BTreeMap",
                make::btree_map_with_size(
                    make::arbitrary::<i32>(),
                    make::arbitrary::<i32>(),
                    ..MAX_SIZE,
                ),
            );
            prop_smoke(
                src,
                "HashSet",
                make::hash_set_with_size::<_, RandomState>(make::arbitrary::<i32>(), ..MAX_SIZE),
            );
            prop_smoke(
                src,
                "HashMap",
                make::hash_map_with_size::<_, _, RandomState>(
                    make::arbitrary::<i32>(),
                    make::arbitrary::<i32>(),
                    ..MAX_SIZE,
                ),
            );
        });
    }

    #[test]
    fn collection_size_limits() {
        check(|src| {
            let a = src.any_of("a", make::int_in_range(..64usize));
            let b = src.any_of("b", make::int_in_range(..64usize));
            let size = Range::new(a.min(b)..=(a.max(b)));

            let vec = src.any_of("vec", vec_with_size(u8::arbitrary(), size));
            assert!(size.contains(&vec.len()));

            let set = src.any_of("set", btree_set_with_size(u8::arbitrary(), size));
            assert!(size.contains(&set.len()));
            let set_collect = src.any_of(
                "set_collect",
                u8::arbitrary().collect_n::<BTreeSet<_>>(size),
            );
            assert!(size.contains(&set_collect.len()));

            let map = src.any_of(
                "map",
                btree_map_with_size(u8::arbitrary(), u8::arbitrary(), size),
            );
            assert!(size.contains(&map.len()));

            let set: HashSet<u8> = src.any_of("set", hash_set_with_size(u8::arbitrary(), size));
            assert!(size.contains(&set.len()));
            let set_collect =
                src.any_of("set_collect", u8::arbitrary().collect_n::<HashSet<_>>(size));
            assert!(size.contains(&set_collect.len()));

            let map: HashMap<u8, u8> = src.any_of(
                "map",
                hash_map_with_size(u8::arbitrary(), u8::arbitrary(), size),
            );
            assert!(size.contains(&map.len()));
        });
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use crate::{Arbitrary as _, tests::bench_gen_next};

    #[bench]
    fn vec_u8(b: &mut test::Bencher) {
        let g = Vec::<u8>::arbitrary();
        bench_gen_next(b, &g);
    }

    #[bench]
    fn vec_u64(b: &mut test::Bencher) {
        let g = Vec::<u64>::arbitrary();
        bench_gen_next(b, &g);
    }
}
