// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{borrow::Borrow, ops::Deref};

#[cfg(feature = "std")]
use core::cell::Cell;
#[cfg(feature = "std")]
use std::thread_local;

use crate::unwind::panic_assume;

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
            panic_assume(msg)
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
            $crate::__private::panic_assume(stringify!($cond));
        }
    };
}

/// [`dbg`] wrapper that only outputs values for visible test case runs.
///
/// In `no_std`, this macro produces no output.
#[cfg(feature = "std")]
#[macro_export]
macro_rules! vdbg {
    () => {
        if $crate::should_log() {
            ::std::dbg!()
        } else {
            ()
        }
    };
    ($val:expr $(,)?) => {
        if $crate::should_log() {
            ::std::dbg!($val)
        } else {
            $val
        }
    };
}

/// [`dbg`] wrapper that only outputs values for visible test case runs.
///
/// In `no_std`, this macro produces no output.
#[cfg(not(feature = "std"))]
#[macro_export]
macro_rules! vdbg {
    () => {
        ()
    };
    ($val:expr $(,)?) => {
        $val
    };
}

/// [`println`] wrapper that only outputs values for visible test case runs.
///
/// In `no_std`, this macro produces no output.
#[cfg(feature = "std")]
#[macro_export]
macro_rules! vprintln {
    ($($arg:tt)*) => {
        if $crate::should_log() {
            ::std::println!($($arg)*);
        }
    };
}

/// [`println`] wrapper that only outputs values for visible test case runs.
///
/// In `no_std`, this macro produces no output.
#[cfg(not(feature = "std"))]
#[macro_export]
macro_rules! vprintln {
    ($($arg:tt)*) => {
        if false {
            let _ = ::core::format_args!($($arg)*);
        }
    };
}

/// [`eprintln`] wrapper that only outputs values for visible test case runs.
///
/// In `no_std`, this macro produces no output.
#[cfg(feature = "std")]
#[macro_export]
macro_rules! veprintln {
    ($($arg:tt)*) => {
        if $crate::should_log() {
            ::std::eprintln!($($arg)*);
        }
    };
}

/// [`eprintln`] wrapper that only outputs values for visible test case runs.
///
/// In `no_std`, this macro produces no output.
#[cfg(not(feature = "std"))]
#[macro_export]
macro_rules! veprintln {
    ($($arg:tt)*) => {
        if false {
            let _ = ::core::format_args!($($arg)*);
        }
    };
}

#[cfg(feature = "std")]
thread_local! {
    // `None` means no Source is active, in which case debug output behaves normally.
    static DEBUG_OUTPUT_SHOULD_LOG: Cell<Option<bool>> = const { Cell::new(None) };
}

#[derive(Debug)]
pub(crate) struct DebugOutputGuard {
    #[cfg(feature = "std")]
    previous: Option<bool>,
}

impl DebugOutputGuard {
    pub(crate) fn new(should_log: bool) -> Self {
        #[cfg(feature = "std")]
        {
            let previous = DEBUG_OUTPUT_SHOULD_LOG.replace(Some(should_log));
            Self { previous }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = should_log;
            Self {}
        }
    }
}

impl Drop for DebugOutputGuard {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        let _ = DEBUG_OUTPUT_SHOULD_LOG.try_with(|value| value.set(self.previous));
    }
}

/// Determine whether debug output should be emitted for the current test-case run.
///
/// By default, this is true only for the final failing run. The state is thread-local and defaults
/// to true outside property execution. In `no_std`, this always returns true.
#[must_use]
pub fn should_log() -> bool {
    #[cfg(feature = "std")]
    {
        DEBUG_OUTPUT_SHOULD_LOG
            .try_with(|value| value.get().unwrap_or(true))
            .unwrap_or(true)
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
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

#[cfg(all(feature = "regex", feature = "std"))]
pub(crate) fn read_lock_no_poison<T>(
    m: &std::sync::RwLock<T>,
) -> std::sync::RwLockReadGuard<'_, T> {
    match m.read() {
        Ok(guard) => guard,
        Err(err) => err.into_inner(),
    }
}

#[cfg(all(feature = "regex", feature = "std"))]
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
