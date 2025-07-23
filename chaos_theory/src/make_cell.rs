// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    cell::{Cell, OnceCell, RefCell, UnsafeCell},
    fmt::Debug,
};

use crate::{Arbitrary, Generator};

impl<T: Arbitrary> Arbitrary for UnsafeCell<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        unsafe_cell(T::arbitrary())
    }
}

impl<T: Arbitrary + Copy> Arbitrary for Cell<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        cell(T::arbitrary())
    }
}

impl<T: Arbitrary> Arbitrary for RefCell<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        ref_cell(T::arbitrary())
    }
}

impl<T: Arbitrary> Arbitrary for OnceCell<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        once_cell(T::arbitrary())
    }
}

#[derive(Debug)]
struct UnsafeCell_<G> {
    elem: G,
}

impl<G: Generator> Generator for UnsafeCell_<G> {
    type Item = UnsafeCell<G::Item>;

    fn next(&self, src: &mut crate::SourceRaw, _example: Option<&Self::Item>) -> Self::Item {
        // `UnsafeCell` read access is unsafe, so we can't look inside the example.
        let v = self.elem.next(src, None);
        UnsafeCell::new(v)
    }
}

#[derive(Debug)]
struct Cell_<G> {
    elem: G,
}

impl<G: Generator<Item: Copy>> Generator for Cell_<G> {
    type Item = Cell<G::Item>;

    fn next(&self, src: &mut crate::SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example = example.map(Cell::get);
        let v = self.elem.next(src, example.as_ref());
        Cell::new(v)
    }
}

#[derive(Debug)]
struct RefCell_<G> {
    elem: G,
}

impl<G: Generator> Generator for RefCell_<G> {
    type Item = RefCell<G::Item>;

    fn next(&self, src: &mut crate::SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let v = if let Some(example) = example.and_then(|e| e.try_borrow().ok()) {
            self.elem.next(src, Some(&*example))
        } else {
            self.elem.next(src, None)
        };
        RefCell::new(v)
    }
}

#[derive(Debug)]
struct OnceCell_<G> {
    elem: G,
}

impl<G: Generator> Generator for OnceCell_<G> {
    type Item = OnceCell<G::Item>;

    fn next(&self, src: &mut crate::SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example = example.and_then(OnceCell::get);
        let v = self.elem.next(src, example);
        let c = OnceCell::new();
        c.set(v).expect("internal error: can't set empty OnceCell");
        c
    }
}

/// Create an [`UnsafeCell`] generator.
pub fn unsafe_cell<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = UnsafeCell<T>>
where
    T: Debug,
{
    UnsafeCell_ { elem }
}

/// Create a [`Cell`] generator.
pub fn cell<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = Cell<T>>
where
    T: Debug + Copy,
{
    Cell_ { elem }
}

/// Create a [`RefCell`] generator.
pub fn ref_cell<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = RefCell<T>>
where
    T: Debug,
{
    RefCell_ { elem }
}

/// Create a [`OnceCell`] generator.
// TODO: allow_uninit
pub fn once_cell<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = OnceCell<T>>
where
    T: Debug,
{
    OnceCell_ { elem }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        check,
        make::int_in_range,
        tests::{any_assert_valid_tape, prop_smoke},
    };

    #[test]
    fn cell_smoke() {
        check(|src| {
            let g = int_in_range::<i32>(..);

            // `prop_smoke` does not work because `UnsafeCell` does not implement `PartialEq`.
            let _ = any_assert_valid_tape(src, "UnsafeCell", unsafe_cell(&g));

            prop_smoke(src, "Cell", cell(&g));
            prop_smoke(src, "RefCell", ref_cell(&g));
            prop_smoke(src, "OnceCell", once_cell(&g));
        });
    }
}
