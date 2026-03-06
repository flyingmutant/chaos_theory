// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::{
    boxed::Box,
    string::{String, ToString as _},
};
use core::{any::Any, fmt::Display};

#[cfg(feature = "std")]
#[path = "unwind_std.rs"]
mod std_impl;

// We have to resort to string prefix hacks, because using custom wrapper type with `panic_any`
// results in a panic message just saying "Box<dyn Any>" (from `PanicHookInfo::payload_as_str`).
pub(crate) const ASSUME_FAILED_PREFIX: &str = "[chaos_theory] assumption failed: ";

#[doc(hidden)]
#[track_caller]
pub fn __panic_assume(msg: &str) -> ! {
    panic!("{ASSUME_FAILED_PREFIX}{msg}");
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PanicInfo {
    pub invalid_data: bool,
    pub message: String,
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl PanicInfo {
    pub(crate) fn same_location(&self, other: &Self) -> bool {
        (&self.file, self.line, self.column) == (&other.file, other.line, other.column)
    }
}

#[cfg(feature = "std")]
pub(crate) fn catch_silent<T, U>(func: impl FnOnce(T) -> U, arg: T) -> Result<U, PanicInfo> {
    std_impl::catch_silent(func, arg)
}

#[cfg(not(feature = "std"))]
pub(crate) fn catch_silent<T, U>(func: impl FnOnce(T) -> U, arg: T) -> Result<U, PanicInfo> {
    Ok((func)(arg))
}

pub(crate) fn panic_message(e: Box<dyn Any + Send>) -> (String, bool) {
    let mut s = match e.downcast::<String>() {
        Ok(s) => *s,
        Err(e) => match e.downcast::<&str>() {
            Ok(s) => (*s).into(),
            Err(e) => {
                if let Ok(d) = e.downcast::<&dyn Display>() {
                    (*d).to_string()
                } else {
                    "<panic with unexpected payload>".into()
                }
            }
        },
    };
    if s.trim().is_empty() {
        s = "<panic with empty payload>".into();
    }
    let is_assume = s.starts_with(ASSUME_FAILED_PREFIX);
    (s, is_assume)
}

#[doc(hidden)]
pub const __ASSUME_FAILED_PREFIX: &str = ASSUME_FAILED_PREFIX;

#[doc(hidden)]
#[must_use]
pub fn __panic_message(err: Box<dyn Any + Send>) -> (String, bool) {
    panic_message(err)
}

#[doc(hidden)]
#[cfg(feature = "std")]
pub fn __catch_silent<T>(func: impl FnOnce() -> T) -> Result<T, Box<dyn Any + Send>> {
    std_impl::__catch_silent(func)
}

#[doc(hidden)]
#[cfg(not(feature = "std"))]
pub fn __catch_silent<T>(func: impl FnOnce() -> T) -> Result<T, Box<dyn Any + Send>> {
    Ok((func)())
}

#[cfg(test)]
mod tests {
    use super::{__panic_assume, catch_silent};

    #[test]
    fn assume_invalid_data() {
        let Err(err) = catch_silent(__panic_assume, "hello");
        assert!(err.invalid_data);
    }
}
