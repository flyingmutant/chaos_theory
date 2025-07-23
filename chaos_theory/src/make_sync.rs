// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{fmt::Debug, marker::PhantomData, ops::RangeBounds};
use std::sync::{Barrier, Condvar, Mutex, Once, OnceLock, RwLock, mpsc};

use crate::{
    Arbitrary, Generator,
    make::{from_fn, size},
    range::SizeRange,
};

impl<T: Arbitrary> Arbitrary for OnceLock<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        once_lock(T::arbitrary())
    }
}

impl Arbitrary for Once {
    fn arbitrary() -> impl Generator<Item = Self> {
        from_fn(|_src, _example| Self::new())
    }
}

impl Arbitrary for Barrier {
    fn arbitrary() -> impl Generator<Item = Self> {
        barrier(..)
    }
}

impl Arbitrary for Condvar {
    fn arbitrary() -> impl Generator<Item = Self> {
        from_fn(|_src, _example| Self::new())
    }
}

impl<T: Arbitrary> Arbitrary for Mutex<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        mutex(T::arbitrary())
    }
}

impl<T: Arbitrary> Arbitrary for RwLock<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        rw_lock(T::arbitrary())
    }
}

impl<T> Arbitrary for (mpsc::Sender<T>, mpsc::Receiver<T>) {
    fn arbitrary() -> impl Generator<Item = Self> {
        from_fn(|_src, _example| mpsc::channel())
    }
}

impl<T> Arbitrary for (mpsc::SyncSender<T>, mpsc::Receiver<T>) {
    fn arbitrary() -> impl Generator<Item = Self> {
        mpsc_sync_channel(..)
    }
}

#[derive(Debug)]
struct OnceLock_<G> {
    elem: G,
}

impl<G: Generator> Generator for OnceLock_<G> {
    type Item = OnceLock<G::Item>;

    fn next(&self, src: &mut crate::SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example = example.and_then(OnceLock::get);
        let v = self.elem.next(src, example);
        let c = OnceLock::new();
        c.set(v).expect("internal error: can't set empty OnceLock");
        c
    }
}

#[derive(Debug)]
struct Barrier_ {
    n: SizeRange,
}

impl Generator for Barrier_ {
    type Item = Barrier;

    fn next(&self, src: &mut crate::SourceRaw, _example: Option<&Self::Item>) -> Self::Item {
        let n = size(self.n).next(src, None); // can't get num threads from a barrier
        Barrier::new(n)
    }
}

#[derive(Debug)]
struct Mutex_<G> {
    elem: G,
}

impl<G: Generator> Generator for Mutex_<G> {
    type Item = Mutex<G::Item>;

    fn next(&self, src: &mut crate::SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let v = if let Some(example) = example.and_then(|e| e.try_lock().ok()) {
            self.elem.next(src, Some(&*example))
        } else {
            self.elem.next(src, None)
        };
        Mutex::new(v)
    }
}

#[derive(Debug)]
struct RwLock_<G> {
    elem: G,
}

impl<G: Generator> Generator for RwLock_<G> {
    type Item = RwLock<G::Item>;

    fn next(&self, src: &mut crate::SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let v = if let Some(example) = example.and_then(|e| e.try_read().ok()) {
            self.elem.next(src, Some(&*example))
        } else {
            self.elem.next(src, None)
        };
        RwLock::new(v)
    }
}

struct MpscSyncChannel_<T> {
    bound: SizeRange,
    _marker: PhantomData<T>,
}

impl<T> Debug for MpscSyncChannel_<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MpscSyncChannel_")
            .field("bound", &self.bound)
            .finish()
    }
}

impl<T> Generator for MpscSyncChannel_<T> {
    type Item = (mpsc::SyncSender<T>, mpsc::Receiver<T>);

    fn next(&self, src: &mut crate::SourceRaw, _example: Option<&Self::Item>) -> Self::Item {
        let n = size(self.bound).next(src, None); // can't get bound from a sender or receiver
        mpsc::sync_channel(n)
    }
}

/// Create a [`OnceLock`] generator.
// TODO: allow_uninit
pub fn once_lock<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = OnceLock<T>>
where
    T: Debug,
{
    OnceLock_ { elem }
}

/// Create a [`Barrier`] generator.
pub fn barrier(n: impl RangeBounds<usize>) -> impl Generator<Item = Barrier> {
    Barrier_ {
        n: SizeRange::new(n),
    }
}

/// Create a [`Mutex`] generator.
pub fn mutex<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = Mutex<T>>
where
    T: Debug,
{
    Mutex_ { elem }
}

/// Create a [`RwLock`] generator.
pub fn rw_lock<T>(elem: impl Generator<Item = T>) -> impl Generator<Item = RwLock<T>>
where
    T: Debug,
{
    RwLock_ { elem }
}

/// Create a ([`mpsc::SyncSender`], [`mpsc::Receiver`]) generator.
pub fn mpsc_sync_channel<T>(
    bound: impl RangeBounds<usize>,
) -> impl Generator<Item = (mpsc::SyncSender<T>, mpsc::Receiver<T>)> {
    MpscSyncChannel_ {
        bound: SizeRange::new(bound),
        _marker: PhantomData,
    }
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
    fn sync_smoke() {
        check(|src| {
            let g = int_in_range::<i32>(..);

            prop_smoke(src, "OnceLock", once_lock(&g));

            // `prop_smoke` does not work because these types do not implement `PartialEq`.
            let _ = any_assert_valid_tape(src, "Barrier", barrier(..));
            let _ = any_assert_valid_tape(src, "Mutex", mutex(&g));
            let _ = any_assert_valid_tape(src, "RwLock", rw_lock(&g));
            let _ = any_assert_valid_tape(src, "Barrier", mpsc_sync_channel::<i32>(..));
        });
    }
}
