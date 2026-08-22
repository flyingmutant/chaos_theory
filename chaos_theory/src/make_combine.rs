// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::vec::Vec;
use core::{fmt::Debug, marker::PhantomData, num::NonZero};

use crate::{Generator, OptionExt as _, Source, SourceEx, env::Tweak};

const FROM_NEXT_SOME_TOO_MUCH: &str = "from_next_some function produced too many None values";

struct FromNext<F, T> {
    next: F,
    _marker: PhantomData<T>,
}

impl<F, T> Debug for FromNext<F, T>
where
    F: Fn(&mut SourceEx, Option<&T>) -> T,
    T: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("FromNext")
            .field(&core::any::type_name::<T>())
            .finish()
    }
}

impl<F, T> Generator for FromNext<F, T>
where
    F: Fn(&mut SourceEx, Option<&T>) -> T,
    T: Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        (self.next)(src, example)
    }
}

struct FromNextSome<F, T> {
    next: F,
    _marker: PhantomData<T>,
}

impl<F, T> Debug for FromNextSome<F, T>
where
    F: Fn(&mut SourceEx, Option<&T>) -> Option<T>,
    T: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("FromNextSome")
            .field(&core::any::type_name::<T>())
            .finish()
    }
}

impl<F, T> Generator for FromNextSome<F, T>
where
    F: Fn(&mut SourceEx, Option<&T>) -> Option<T>,
    T: Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let v = src.find("<from-next-some>", example, &self.next);
        v.assume_some_msg(FROM_NEXT_SOME_TOO_MUCH)
    }
}

#[derive(Debug)]
struct Just<T> {
    value: T,
}

impl<T: Debug + Clone> Generator for Just<T> {
    type Item = T;

    fn next(&self, _src: &mut SourceEx, _example: Option<&Self::Item>) -> Self::Item {
        self.value.clone()
    }
}

#[derive(Debug)]
struct OneOfOwned<T> {
    elements: Vec<T>,
}

impl<T: Debug + Clone> Generator for OneOfOwned<T> {
    type Item = T;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        OneOf {
            elements: &self.elements,
        }
        .next(src, example)
    }
}

#[derive(Debug)]
struct OneOf<'elems, T> {
    elements: &'elems [T],
}

impl<T: Debug + Clone> Generator for OneOf<'_, T> {
    type Item = T;

    fn next(&self, src: &mut SourceEx, _example: Option<&Self::Item>) -> Self::Item {
        // TODO: search for an example in the elements?
        let e_ix = src
            .as_mut()
            .choose_index(self.elements.len(), None, Tweak::None);
        self.elements[e_ix].clone()
    }
}

#[derive(Debug)]
struct MixOfOwned<G> {
    gens: Vec<G>,
    n: NonZero<usize>,
}

impl<G: Generator> Generator for MixOfOwned<G> {
    type Item = G::Item;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        MixOf {
            gens: &self.gens,
            n: self.n,
        }
        .next(src, example)
    }
}

#[derive(Debug)]
struct MixOf<'gens, G> {
    gens: &'gens [G],
    n: NonZero<usize>,
}

impl<G: Generator> Generator for MixOf<'_, G> {
    type Item = G::Item;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        src.select(
            "<mixof-gen>",
            None,
            self.n,
            |_| "",
            |src, _variant, ix| {
                let g = &self.gens[ix];
                g.next(src, example)
            },
        )
    }
}

// TODO: recursive

/// Create a generator that uses the provided closure to construct items.
///
/// Consider [`from_next`] if you want to support example-guided generation.
pub fn from_fn<T: Debug>(func: impl Fn(&mut Source) -> T) -> impl Generator<Item = T> {
    from_next(move |src, _example| {
        let mut src = Source::new(src.as_mut());
        func(&mut src)
    })
}

/// Create a generator that retries the closure until it constructs a `Some` item.
///
/// If the closure returns `None` too often, the whole test case is marked invalid.
///
/// Consider [`from_next_some`] if you want to support example-guided generation.
pub fn from_fn_some<T: Debug>(func: impl Fn(&mut Source) -> Option<T>) -> impl Generator<Item = T> {
    from_next_some(move |src, _example| {
        let mut src = Source::new(src.as_mut());
        func(&mut src)
    })
}

/// Create a generator that uses the provided closure to construct items, taking examples into account.
///
/// Prefer [`from_fn`] if you don't need to support example-guided generation.
pub fn from_next<T: Debug>(
    next: impl Fn(&mut SourceEx, Option<&T>) -> T,
) -> impl Generator<Item = T> {
    FromNext {
        next,
        _marker: PhantomData,
    }
}

/// Create a generator that retries the closure until it constructs a `Some` item,
/// taking examples into account.
///
/// If the closure returns `None` too often, the whole test case is marked invalid.
///
/// Don't forget to filter the examples according to the same predicate you use to construct `Some`.
///
/// Prefer [`from_fn_some`] if you don't need to support example-guided generation.
// TODO: do we really want/need to filter examples?
pub fn from_next_some<T: Debug>(
    next: impl Fn(&mut SourceEx, Option<&T>) -> Option<T>,
) -> impl Generator<Item = T> {
    FromNextSome {
        next,
        _marker: PhantomData,
    }
}

/// Create a generator that always yields the same value.
///
/// Use [`one_of`] when you need to choose between multiple values.
pub fn just<T: Debug + Clone>(value: T) -> impl Generator<Item = T> {
    Just { value }
}

/// Create a generator that yields one of the specified values.
///
/// To choose between multiple generators (not values) use [`Generator::or`] or [`mix_of`].
///
/// # Panics
///
/// `one_of` panics when elements slice is empty.
pub fn one_of<T: Debug + Clone>(elements: &[T]) -> impl Generator<Item = T> {
    assert!(!elements.is_empty(), "no elements provided");
    OneOf { elements }
}

/// Create a generator that yields one of the specified values.
///
/// To choose between multiple generators (not values) use [`Generator::or`] or [`mix_of`].
///
/// # Panics
///
/// `one_of_owned` panics when elements vector is empty.
pub fn one_of_owned<T: Debug + Clone>(elements: Vec<T>) -> impl Generator<Item = T> {
    assert!(!elements.is_empty(), "no elements provided");
    OneOfOwned { elements }
}

/// Create a generator that yields values from all of the supplied generators.
///
/// When possible, prefer using the more efficient [`Generator::or`] instead.
///
/// To choose between multiple values (not generators) use [`one_of`].
///
/// # Panics
///
/// `mix_of` panics when generators vector is empty.
pub fn mix_of<G: Generator>(gens: &[G]) -> impl Generator<Item = G::Item> {
    let n = NonZero::new(gens.len()).expect("no generators provided");
    MixOf { gens, n }
}

/// Create a generator that yields values from all of the supplied generators.
///
/// When possible, prefer using the more efficient [`Generator::or`] instead.
///
/// To choose between multiple values (not generators) use [`one_of`].
///
/// # Panics
///
/// `mix_of_owned` panics when generators vector is empty.
pub fn mix_of_owned<G: Generator>(gens: Vec<G>) -> impl Generator<Item = G::Item> {
    let n = NonZero::new(gens.len()).expect("no generators provided");
    MixOfOwned { gens, n }
}
