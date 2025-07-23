// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{fmt::Debug, marker::PhantomData, num::NonZero};

use crate::{Generator, OptionExt as _, SourceRaw, Tweak};

const FROM_FN_ASSUME_TOO_MUCH: &str = "from_fn_assume function produced too many None values";

struct FromFn<F, T> {
    func: F,
    _marker: PhantomData<T>,
}

impl<F, T> Debug for FromFn<F, T>
where
    F: Fn(&mut SourceRaw, Option<&T>) -> T,
    T: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("FromFn")
            .field(&core::any::type_name::<T>())
            .finish()
    }
}

impl<F, T> Generator for FromFn<F, T>
where
    F: Fn(&mut SourceRaw, Option<&T>) -> T,
    T: Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        (self.func)(src, example)
    }
}

struct FromFnAssume<F, T> {
    func: F,
    _marker: PhantomData<T>,
}

impl<F, T> Debug for FromFnAssume<F, T>
where
    F: Fn(&mut SourceRaw, Option<&T>) -> Option<T>,
    T: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("FromFnAssume")
            .field(&core::any::type_name::<T>())
            .finish()
    }
}

impl<F, T> Generator for FromFnAssume<F, T>
where
    F: Fn(&mut SourceRaw, Option<&T>) -> Option<T>,
    T: Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let v = src.find("<from-fn-assume>", example, &self.func);
        v.assume_some_msg(FROM_FN_ASSUME_TOO_MUCH)
    }
}

#[derive(Debug)]
struct Just<T> {
    value: T,
}

impl<T: Debug + Clone> Generator for Just<T> {
    type Item = T;

    fn next(&self, _src: &mut SourceRaw, _example: Option<&Self::Item>) -> Self::Item {
        self.value.clone()
    }
}

#[derive(Debug)]
struct OneOfOwned<T> {
    elements: Vec<T>,
}

impl<T: Debug + Clone> Generator for OneOfOwned<T> {
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
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

    fn next(&self, src: &mut SourceRaw, _example: Option<&Self::Item>) -> Self::Item {
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

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
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

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
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
pub fn from_fn<T: Debug>(
    func: impl Fn(&mut SourceRaw, Option<&T>) -> T,
) -> impl Generator<Item = T> {
    FromFn {
        func,
        _marker: PhantomData,
    }
}

/// Create a filtering generator that yields `Some` items constructed by the provided closure.
///
/// Don't forget to filter the examples according to the same predicate you use to construct `Some`.
pub fn from_fn_assume<T: Debug>(
    func: impl Fn(&mut SourceRaw, Option<&T>) -> Option<T>,
) -> impl Generator<Item = T> {
    FromFnAssume {
        func,
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
