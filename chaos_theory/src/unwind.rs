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
#[doc(hidden)]
pub const ASSUME_FAILED_PREFIX: &str = "[chaos_theory] assumption failed: ";
pub(crate) const DETERMINISM_FAILED_PREFIX: &str = "[chaos_theory] determinism check failed: ";

#[doc(hidden)]
#[track_caller]
pub fn panic_assume(msg: &str) -> ! {
    panic!("{ASSUME_FAILED_PREFIX}{msg}");
}

pub(crate) fn panic_determinism(msg: impl Display) -> ! {
    panic!("{DETERMINISM_FAILED_PREFIX}{msg}");
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PanicInfo {
    pub invalid_data: bool,
    pub determinism_failure: bool,
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
pub(crate) fn catch_silent_info<T, U>(func: impl FnOnce(T) -> U, arg: T) -> Result<U, PanicInfo> {
    std_impl::catch_silent_info(func, arg)
}

#[cfg(not(feature = "std"))]
pub(crate) fn catch_silent_info<T, U>(func: impl FnOnce(T) -> U, arg: T) -> Result<U, PanicInfo> {
    Ok((func)(arg))
}

#[doc(hidden)]
#[must_use]
pub fn panic_message(e: Box<dyn Any + Send>) -> (String, bool, bool) {
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
    let is_determinism_failure = s.starts_with(DETERMINISM_FAILED_PREFIX);
    (s, is_assume, is_determinism_failure)
}

#[cfg(feature = "std")]
#[doc(hidden)]
pub fn catch_silent<T>(func: impl FnOnce() -> T) -> Result<T, Box<dyn Any + Send>> {
    std_impl::catch_silent(func)
}

#[cfg(not(feature = "std"))]
#[doc(hidden)]
pub fn catch_silent<T>(func: impl FnOnce() -> T) -> Result<T, Box<dyn Any + Send>> {
    Ok((func)())
}

#[cfg(test)]
mod tests {
    use super::{catch_silent_info, panic_assume};

    #[test]
    fn assume_invalid_data() {
        let Err(err) = catch_silent_info(panic_assume, "hello");
        assert!(err.invalid_data);
    }
}
