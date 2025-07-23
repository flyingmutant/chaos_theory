// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{borrow::Borrow, ops::Deref};

use crate::__panic_assume;

const ASSUME_SOME_FAILED_MSG: &str = "OptionExt::assume_some failed";

/// Utility trait to give [`Option`] an [`OptionExt::assume_some`] helper.
pub trait OptionExt<T>: Sized {
    /// [`assume`](crate::assume) that `Option` value is `Some` and unwrap it.
    #[track_caller]
    fn assume_some(self) -> T {
        self.assume_some_msg(ASSUME_SOME_FAILED_MSG)
    }

    /// [`assume`](crate::assume) that `Option` value is `Some` and unwrap it.
    #[track_caller]
    fn assume_some_msg(self, msg: &str) -> T;
}

impl<T> OptionExt<T> for Option<T> {
    fn assume_some_msg(self, msg: &str) -> T {
        if let Some(v) = self {
            v
        } else {
            __panic_assume(msg)
        }
    }
}

/// Mark the current test case as invalid if the expression is false.
///
/// `assume` should be used carefully, as too many invalid test cases
/// will make [`Env::check`](crate::Env::check) panic because it is unable
/// to generate enough valid test cases to check the property.
///
/// When possible, prefer [`Generator::filter`](crate::Generator::filter)
/// to `assume`, and prefer generators that always produce valid values to `filter`.
#[macro_export]
macro_rules! assume {
    ($cond:expr) => {
        if !$cond {
            $crate::__panic_assume(stringify!($cond));
        }
    };
}

/// [`dbg`] wrapper that checks [`Source::should_log`](crate::Source::should_log).
///
/// First argument should be a [`Source`](crate::Source) reference, the rest are forwarded to `dbg`.
#[macro_export]
macro_rules! vdbg {
    ($src:expr, $($arg:tt)*) => {
        if $src.should_log() {
            ::std::dbg!($($arg)*)
        } else {
            $($arg)*
        }
    };
}

/// [`println`] wrapper that checks [`Source::should_log`](crate::Source::should_log).
///
/// First argument should be a [`Source`](crate::Source) reference, the rest are forwarded to `println`.
#[macro_export]
macro_rules! vprintln {
    ($src:expr, $($arg:tt)*) => {
        if $src.should_log() {
            ::std::println!($($arg)*);
        }
    };
}

/// [`eprintln`] wrapper that checks [`Source::should_log`](crate::Source::should_log).
///
/// First argument should be a [`Source`](crate::Source) reference, the rest are forwarded to `eprintln`.
#[macro_export]
macro_rules! veprintln {
    ($src:expr, $($arg:tt)*) => {
        if $src.should_log() {
            ::std::eprintln!($($arg)*);
        }
    };
}

/// Type that represents either owned or borrowed values.
#[derive(Debug)]
pub enum MaybeOwned<'a, T: 'a> {
    /// Owned value.
    Owned(T),
    /// Borrowed value.
    Borrowed(&'a T),
}

impl<T> Borrow<T> for MaybeOwned<'_, T> {
    fn borrow(&self) -> &T {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v,
        }
    }
}

impl<T> Deref for MaybeOwned<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.borrow()
    }
}

#[cfg(feature = "regex")]
pub(crate) fn read_lock_no_poison<T>(
    m: &std::sync::RwLock<T>,
) -> std::sync::RwLockReadGuard<'_, T> {
    match m.read() {
        Ok(guard) => guard,
        Err(err) => err.into_inner(),
    }
}

#[cfg(feature = "regex")]
pub(crate) fn write_lock_no_poison<T>(
    m: &std::sync::RwLock<T>,
) -> std::sync::RwLockWriteGuard<'_, T> {
    match m.write() {
        Ok(guard) => guard,
        Err(err) => err.into_inner(),
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use core::panic::Location;

    #[track_caller]
    fn tracked_caller() -> &'static Location<'static> {
        Location::caller()
    }

    #[bench]
    fn caller(b: &mut test::Bencher) {
        b.iter(tracked_caller);
    }
}
