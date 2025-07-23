// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::{rc::Rc, sync::Arc};
use core::{
    any::Any,
    borrow::Borrow,
    fmt::Debug,
    marker::PhantomData,
    num::NonZero,
    ops::{Deref, RangeBounds},
};

use crate::{
    Effect, MaybeOwned, OptionExt as _, SeedTapeReplayScope, SourceRaw, USE_SEED_PROB,
    range::SizeRange,
};

const FILTER_ASSUME_TOO_MUCH: &str = "filter_assume predicate rejected too many generated values";
pub(crate) const UNABLE_GENERATE_UNIQUE: &str =
    "unable to generate specified number of unique values";

/// A trait for types that describe composable generation of values.
#[must_use]
pub trait Generator: Debug {
    /// The type of the generated values.
    type Item: Debug;

    /// Generate a new item, optionally using the provided example.
    ///
    /// **Important**: `next` should not be called directly, as that can violate internal `chaos_theory` invariants.
    /// [`Source::any_of`](crate::Source::any_of) or [`SourceRaw::any_of`] methods should be used instead.
    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item;

    /// Create a generator which uses the seeds to bias the generated items.
    ///
    /// If `mutate_seeds` is false, seed values will always be used as-is. Otherwise,
    /// randomly mutated versions of seeds will be used together with non-mutated ones.
    fn seeded(self, seeds: &[Self::Item], mutate_seeds: bool) -> impl Generator<Item = Self::Item>
    where
        Self: Sized,
    {
        Seeded {
            gen_: self,
            seeds,
            mutate_seeds,
        }
    }

    /// Create a generator which uses the seeds to bias the generated items.
    ///
    /// If `mutate_seeds` is false, seed values will always be used as-is. Otherwise,
    /// randomly mutated versions of seeds will be used together with non-mutated ones.
    fn seeded_owned(
        self,
        seeds: Vec<Self::Item>,
        mutate_seeds: bool,
    ) -> impl Generator<Item = Self::Item>
    where
        Self: Sized,
    {
        SeededOwned {
            gen_: self,
            seeds,
            mutate_seeds,
        }
    }

    /// Create a generator which uses a predicate to filter the generated items.
    ///
    /// `filter` will return `None` when it is unable to satisfy the predicate in
    /// some number of tries.
    ///
    /// Use `filter` when you can gracefully react to failure to satisfy the predicate
    /// (for example, by returning [`Effect::Noop`] to retry a [`Source::repeat`](crate::Source::repeat)
    /// step). Otherwise, use [`Generator::filter_assume`].
    fn filter<F>(self, predicate: F) -> impl Generator<Item = Option<Self::Item>>
    where
        F: Fn(&Self::Item) -> bool,
        Self: Sized,
    {
        Filter {
            gen_: self,
            pred: predicate,
        }
    }

    /// Create a generator which uses a predicate to filter the generated items.
    ///
    /// `filter_assume` should be used carefully, because predicates that reject too many items
    /// will lead to [`Env::check`](crate::Env::check) panicing due to being unable to
    /// construct enough valid test cases.
    ///
    /// When possible, prefer [`Generator::filter`] or generators that always produce
    /// valid values to `filter_assume`.
    fn filter_assume<F>(self, predicate: F) -> impl Generator<Item = Self::Item>
    where
        F: Fn(&Self::Item) -> bool,
        Self: Sized,
    {
        FilterAssume {
            gen_: self,
            pred: predicate,
        }
    }

    /// Create a generator of a collections of items.
    fn collect<C>(self) -> impl Generator<Item = C>
    where
        Self: Sized,
        C: Debug + Default + Extend<Self::Item>,
        for<'a> &'a C: IntoIterator<Item = &'a Self::Item, IntoIter: ExactSizeIterator>,
    {
        self.collect_n(..)
    }

    /// Create a generator of a collections of items, with the specified size.
    fn collect_n<C>(self, n: impl RangeBounds<usize>) -> impl Generator<Item = C>
    where
        Self: Sized,
        C: Debug + Default + Extend<Self::Item>,
        for<'a> &'a C: IntoIterator<Item = &'a Self::Item, IntoIter: ExactSizeIterator>,
    {
        Collect {
            elem: self,
            size: SizeRange::new(n),
            _marker: PhantomData,
        }
    }

    /// Create a generator which yields items from either this one or `other`.
    ///
    /// Use [`mix_of`](crate::make::mix_of) when the number of the generators to combine is variable at runtime.
    fn or<G>(self, other: G) -> impl Generator<Item = Self::Item>
    where
        G: Generator<Item = Self::Item>,
        Self: Sized,
    {
        Or {
            gens: (self, other),
        }
    }

    /// Create a generator that applies a closure to all items.
    ///
    /// When possible, prefer using [`Generator::map_reversible`] or other specialized
    /// `Generator::map_` methods instead.
    fn map<F, U>(self, f: F) -> impl Generator<Item = U>
    where
        F: Fn(Self::Item) -> U,
        U: Debug,
        Self: Sized,
    {
        self.map_reversible(f, |_| None)
    }

    /// Create a generator that applies a closure to all items, and specifies the reverse transformation.
    fn map_reversible<F, R, U>(self, f: F, r: R) -> impl Generator<Item = U>
    where
        F: Fn(Self::Item) -> U,
        U: Debug,
        R: Fn(&U) -> Option<MaybeOwned<'_, Self::Item>>,
        Self: Sized,
    {
        Map {
            gen_: self,
            func: f,
            rev: r,
        }
    }

    /// Create a generator of type-erased `Box<dyn Any>` items.
    fn map_any(self) -> impl Generator<Item = Box<dyn Any>>
    where
        Self: Sized,
        Self::Item: 'static,
    {
        self.map_reversible(
            |v| Box::new(v) as Box<dyn Any>,
            |u| u.downcast_ref().map(MaybeOwned::Borrowed),
        )
    }

    /// Create a generator of items for which `Into` + `Deref` conversions exist.
    fn map_into_deref<U>(self) -> impl Generator<Item = U>
    where
        Self: Sized,
        Self::Item: Into<U>,
        U: Debug + Deref<Target = Self::Item>,
    {
        self.map_reversible(Into::into, |u| Some(MaybeOwned::Borrowed(&**u)))
    }

    /// Create a generator of items for which `Into` + `Borrow` conversions exist.
    fn map_into_borrow<U>(self) -> impl Generator<Item = U>
    where
        Self: Sized,
        Self::Item: Into<U>,
        U: Debug + Borrow<Self::Item>,
    {
        self.map_reversible(Into::into, |u| Some(MaybeOwned::Borrowed(u.borrow())))
    }

    /// Create a generator of items for which `Into` + `TryFrom` conversions exist.
    fn map_into_try_from<U>(self) -> impl Generator<Item = U>
    where
        Self: Sized,
        Self::Item: Into<U> + TryFrom<U>,
        U: Debug + Copy,
    {
        self.map_reversible(Into::into, |u| {
            Self::Item::try_from(*u).ok().map(MaybeOwned::Owned)
        })
    }

    /// Create a generator that applies a closure to construct another generator, and yield an item from it.
    ///
    /// This operation is sometimes called flatmap.
    fn and_then<'label, F, G, U>(self, f: F) -> impl Generator<Item = U>
    where
        F: Fn(Self::Item) -> (&'label str, G),
        G: Generator<Item = U>,
        U: Debug,
        Self: Sized,
    {
        FlatMap {
            gen_: self,
            func: f,
        }
    }

    /// Convert to a type-erased [`Gen`] instance.
    fn boxed<'a>(self) -> Gen<'a, Self::Item>
    where
        Self: Sized + 'a,
    {
        Gen(Rc::new(self))
    }

    /// Convert to a thread-safe type-erased [`GenShared`] instance.
    fn shared<'a>(self) -> GenShared<'a, Self::Item>
    where
        Self: Send + Sync + Sized + 'a,
    {
        GenShared(Arc::new(self))
    }
}

impl<G: Generator + ?Sized> Generator for &G {
    type Item = G::Item;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        (**self).next(src, example)
    }
}

/// A trait for types that have a default [`Generator`] implementation.
pub trait Arbitrary: Debug + Sized {
    /// Get the default [`Generator`] implementation for type.
    fn arbitrary() -> impl Generator<Item = Self>;
}

/// Type-erased generator of `T`, implemented as a boxed trait object.
///
/// `Gen` implements `Clone` efficiently via reference-counting.
///
/// Use [`Generator::boxed`] to create new instances.
#[must_use]
#[derive(Debug)]
pub struct Gen<'a, T>(Rc<dyn Generator<Item = T> + 'a>);

impl<T> Clone for Gen<'_, T> {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl<T: Debug> Generator for Gen<'_, T> {
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&T>) -> T {
        self.0.next(src, example)
    }

    fn boxed<'b>(self) -> Gen<'b, Self::Item>
    where
        Self: 'b,
    {
        self
    }
}

/// Thread-safe type-erased generator of `T`, implemented as a boxed trait object.
///
/// `GenShared` implements `Clone` efficiently via atomic reference-counting.
///
/// Use [`Generator::shared()`] to create new instances.
#[must_use]
#[derive(Debug)]
pub struct GenShared<'a, T>(Arc<dyn Generator<Item = T> + Send + Sync + 'a>);

impl<T> Clone for GenShared<'_, T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T: Debug> Generator for GenShared<'_, T> {
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&T>) -> T {
        self.0.next(src, example)
    }

    fn shared<'b>(self) -> GenShared<'b, Self::Item>
    where
        Self: 'b,
    {
        self
    }
}

struct SeededOwned<G: Generator> {
    gen_: G,
    seeds: Vec<G::Item>,
    mutate_seeds: bool,
}

impl<G: Generator> Debug for SeededOwned<G> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Seeded")
            .field(&self.gen_)
            .field(&self.mutate_seeds)
            .finish()
    }
}

impl<G: Generator> Generator for SeededOwned<G> {
    type Item = G::Item;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        Seeded {
            gen_: &self.gen_,
            seeds: &self.seeds,
            mutate_seeds: self.mutate_seeds,
        }
        .next(src, example)
    }
}

struct Seeded<'seeds, G: Generator> {
    gen_: G,
    seeds: &'seeds [G::Item],
    mutate_seeds: bool,
}

impl<G: Generator> Debug for Seeded<'_, G> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Seeded")
            .field(&self.gen_)
            .field(&self.mutate_seeds)
            .finish()
    }
}

impl<G: Generator> Generator for Seeded<'_, G> {
    type Item = G::Item;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        if example.is_some() {
            // When example is provided, we ignore the seeds completely and try to reconstruct the example.
            self.gen_.next(src, example)
        } else if self.mutate_seeds {
            let scope = SeedTapeReplayScope::new(src, &self.gen_, self.seeds);
            self.gen_.next(scope.src, None)
        } else {
            // Optimization: no need to create a tape if we will not mutate it.
            let seed = src.as_mut().choose_seed(USE_SEED_PROB, self.seeds);
            self.gen_.next(src, seed)
        }
    }
}

struct Filter<G, F> {
    gen_: G,
    pred: F,
}

impl<G, F> Debug for Filter<G, F>
where
    G: Generator,
    F: Fn(&G::Item) -> bool,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Filter").field(&self.gen_).finish()
    }
}

impl<G, F> Generator for Filter<G, F>
where
    G: Generator,
    F: Fn(&G::Item) -> bool,
{
    type Item = Option<G::Item>;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        src.find(
            "<filter>",
            example.and_then(Option::as_ref),
            |src, example| {
                let example = example.filter(|e| (self.pred)(e));
                let v = self.gen_.next(src, example);
                (self.pred)(&v).then_some(v)
            },
        )
    }
}

struct FilterAssume<G, F> {
    gen_: G,
    pred: F,
}

impl<G, F> Debug for FilterAssume<G, F>
where
    G: Generator,
    F: Fn(&G::Item) -> bool,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("FilterAssume").field(&self.gen_).finish()
    }
}

impl<G, F> Generator for FilterAssume<G, F>
where
    G: Generator,
    F: Fn(&G::Item) -> bool,
{
    type Item = G::Item;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let v = src.find("<filter-assume>", example, |src, example| {
            let example = example.filter(|e| (self.pred)(e));
            let v = self.gen_.next(src, example);
            (self.pred)(&v).then_some(v)
        });
        v.assume_some_msg(FILTER_ASSUME_TOO_MUCH)
    }
}

struct Map<G, F, R> {
    gen_: G,
    func: F,
    rev: R,
}

impl<G, F, T, R> Debug for Map<G, F, R>
where
    G: Generator,
    F: Fn(G::Item) -> T,
    T: Debug,
    R: Fn(&T) -> Option<MaybeOwned<'_, G::Item>>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Map")
            .field(&self.gen_)
            .field(&core::any::type_name::<T>())
            .finish()
    }
}

impl<G, F, T, R> Generator for Map<G, F, R>
where
    G: Generator,
    F: Fn(G::Item) -> T,
    T: Debug,
    R: Fn(&T) -> Option<MaybeOwned<'_, G::Item>>,
{
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        // TODO: we can't go in reverse from &T to &Wrapper(T) (like &Option<T>) without cloning
        let example = example.and_then(|e| (self.rev)(e));
        let arg = self.gen_.next(src, example.as_deref());
        (self.func)(arg)
    }
}

struct FlatMap<G, F> {
    gen_: G,
    func: F,
}

impl<'label, G, F, Q, T> Debug for FlatMap<G, F>
where
    G: Generator,
    F: Fn(G::Item) -> (&'label str, Q),
    Q: Generator<Item = T>,
    T: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("FlatMap")
            .field(&self.gen_)
            .field(&core::any::type_name::<T>())
            .finish()
    }
}

impl<'label, G, F, Q, T> Generator for FlatMap<G, F>
where
    G: Generator,
    F: Fn(G::Item) -> (&'label str, Q),
    Q: Generator<Item = T>,
    T: Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let arg = src.any_of("<flatmap-arg>", &self.gen_, None);
        let (label, gen_) = (self.func)(arg);
        src.any_of(label, gen_, example)
    }
}

trait ExtendOne<A> {
    fn extend_reserve(&mut self, additional: usize);
    fn extend_one(&mut self, item: A);
}

impl<T, A> ExtendOne<A> for T
where
    T: Extend<A>,
{
    fn extend_reserve(&mut self, _additional: usize) {
        // https://github.com/rust-lang/rust/issues/72631
    }

    fn extend_one(&mut self, item: A) {
        <Self as Extend<A>>::extend(self, Some(item));
    }
}

#[derive(Debug)]
struct Collect<G, C> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<C>,
}

impl<G: Generator, C> Generator for Collect<G, C>
where
    C: Debug + Default + ExtendOne<G::Item>,
    for<'a> &'a C: IntoIterator<Item = &'a G::Item, IntoIter: ExactSizeIterator>,
{
    type Item = C;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(IntoIterator::into_iter);
        let res = src.repeat(
            "<collect>",
            example_seq,
            self.size,
            |n| {
                let mut c: C = Default::default();
                c.extend_reserve(n);
                let s = c.into_iter().len();
                (c, s)
            },
            |v: &mut (C, usize), src, example| {
                let elem = self.elem.next(src, example);
                let size_before = v.1;
                v.0.extend_one(elem);
                let size_after = v.0.into_iter().len();
                if size_after == size_before {
                    // We can't mark the step as noop, since `extend_one` replaces the old value in case of duplicates.
                    Effect::Change
                } else {
                    v.1 = size_after;
                    Effect::Success
                }
            },
        );
        res.assume_some_msg(UNABLE_GENERATE_UNIQUE).0
    }
}

macro_rules! define_or {
    ($name: ident, $next: ident, $n: literal, $($params: ident)+, $($ixs: tt)+) => {
        #[derive(Debug)]
        struct $name<$($params, )+>{
            gens: ($($params, )+),
        }

        impl<T: Debug, $($params: Generator<Item = T>, )+> Generator for $name<$($params, )+> {
            type Item = T;

            fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
                let n = NonZero::new($n).expect("internal error: zero-sized Or tuple");
                src.select("<or>", None, n, |_| "", |src, _variant, ix| {
                    match ix {
                        $($ixs => self.gens.$ixs.next(src, example),)+
                        _ => unreachable!(),
                    }
                })
            }

            fn or<_G>(self, other: _G) -> impl Generator<Item = Self::Item>
            where
                _G: Generator<Item = Self::Item>,
                Self: Sized,
            {
                $next{
                    gens: ($(self.gens.$ixs, )+ other),
                }
            }
        }
    };
}

macro_rules! define_or_last {
    ($name: ident, $n: literal, $($params: ident)+, $($ixs: tt)+) => {
        #[derive(Debug)]
        struct $name<$($params, )+>{
            gens: ($($params, )+),
        }

        impl<T: Debug, $($params: Generator<Item = T>, )+> Generator for $name<$($params, )+> {
            type Item = T;

            // We don't override `or`, so we will loop back to 2-case `Or`.
            // This is not ideal, since we now have a non-flat selection,
            // but should be good enough: more than 12-long `or` chains
            // should be extremely unlikely in practice.

            fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
                let n = NonZero::new($n).expect("internal error: zero-sized Or tuple");
                src.select("<or>", None, n, |_| "", |src, _variant, ix| {
                    match ix {
                        $($ixs => self.gens.$ixs.next(src, example),)+
                        _ => unreachable!(),
                    }
                })
            }
        }
    };
}

define_or!(Or,   Or3,  2,  A B,                     0 1);
define_or!(Or3,  Or4,  3,  A B C,                   0 1 2);
define_or!(Or4,  Or5,  4,  A B C D,                 0 1 2 3);
define_or!(Or5,  Or6,  5,  A B C D E,               0 1 2 3 4);
define_or!(Or6,  Or7,  6,  A B C D E F,             0 1 2 3 4 5);
define_or!(Or7,  Or8,  7,  A B C D E F G,           0 1 2 3 4 5 6);
define_or!(Or8,  Or9,  8,  A B C D E F G H,         0 1 2 3 4 5 6 7);
define_or!(Or9,  Or10, 9,  A B C D E F G H I,       0 1 2 3 4 5 6 7 8);
define_or!(Or10, Or11, 10, A B C D E F G H I J,     0 1 2 3 4 5 6 7 8 9);
define_or!(Or11, Or12, 11, A B C D E F G H I J K,   0 1 2 3 4 5 6 7 8 9 10);
define_or_last!( Or12, 12, A B C D E F G H I J K L, 0 1 2 3 4 5 6 7 8 9 10 11);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check;

    #[test]
    fn filter() {
        check(|src| {
            let i = src.any_of("i", u8::arbitrary().filter_assume(|u| *u != 0));
            assert_ne!(i, 0);
        });
    }

    #[test]
    fn map() {
        check(|src| {
            let i = src.any_of("i", u8::arbitrary().map(|u| u.wrapping_mul(2)));
            assert_eq!(i % 2, 0);
        });
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use alloc::collections::BTreeSet;
    use std::collections::HashSet;

    use crate::{Arbitrary as _, Generator as _, tests::bench_gen_next};

    #[bench]
    fn collect_vec_u64(b: &mut test::Bencher) {
        let g = u64::arbitrary().collect::<Vec<_>>();
        bench_gen_next(b, &g);
    }

    #[bench]
    fn collect_btree_set_u64(b: &mut test::Bencher) {
        let g = u64::arbitrary().collect::<BTreeSet<_>>();
        bench_gen_next(b, &g);
    }

    #[bench]
    fn collect_hash_set_u64(b: &mut test::Bencher) {
        let g = u64::arbitrary().collect::<HashSet<_>>();
        bench_gen_next(b, &g);
    }
}
