// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{any::Any, cell::Cell, fmt::Display};
use std::{
    panic::{self, panic_any},
    sync::Once,
    thread_local,
};

// We have to resort to string prefix hacks, because using custom wrapper type with `panic_any`
// results in a panic message just saying "Box<dyn Any>" (from `PanicHookInfo::payload_as_str`).
pub(crate) const ASSUME_FAILED_PREFIX: &str = "[chaos_theory] assumption failed: ";

#[doc(hidden)]
#[track_caller]
pub fn __panic_assume(msg: &str) -> ! {
    panic_any(String::from_iter([ASSUME_FAILED_PREFIX, msg]))
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

pub(crate) fn catch_silent<T, U>(func: impl FnOnce(T) -> U, arg: T) -> Result<U, PanicInfo> {
    let r = __catch_silent(|| (func)(arg));
    r.map_err(|e| {
        let (message, invalid_data) = panic_message(e);
        let (file, line, column) = SilentPanicGuard::take_location();
        PanicInfo {
            invalid_data,
            message,
            file,
            line,
            column,
        }
    })
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
pub fn __catch_silent<T>(func: impl FnOnce() -> T) -> Result<T, Box<dyn Any + Send>> {
    let f = panic::AssertUnwindSafe(|| {
        let _guard = SilentPanicGuard::new();
        (func)()
    });
    panic::catch_unwind(f)
}

type PanicHook = Box<dyn Fn(&panic::PanicHookInfo<'_>) + Send + Sync>;

struct SilentPanicGuard {
    _private: (),
}

impl SilentPanicGuard {
    thread_local! {
        static SILENCE_PANICS: Cell<u32> = const { Cell::new(0) };
        static PANIC_LOCATION: Cell<(String, u32, u32)> = const { Cell::new((String::new(), 0, 0)) };
    }

    fn new() -> Self {
        static PANIC_HOOK_ONCE: Once = Once::new();
        PANIC_HOOK_ONCE.call_once(|| {
            let prev_hook = panic::take_hook();
            // We have taken the previous hook, but not installed ours yet.
            // Can race with other (but not ours, thanks to Once) hook users,
            // unless `update_hook` is stabilized in https://github.com/rust-lang/rust/issues/92649.
            panic::set_hook(Box::new(move |info| {
                Self::hook(&prev_hook, info);
            }));
        });

        Self::SILENCE_PANICS.with(|c| c.update(|v| v + 1));

        Self { _private: () }
    }

    fn hook(prev_hook: &PanicHook, info: &panic::PanicHookInfo) {
        if Self::SILENCE_PANICS.get() > 0 {
            // Remember panic location, without calling the hook chain.
            if let Some(loc) = info.location() {
                Self::PANIC_LOCATION.set((loc.file().into(), loc.line(), loc.column()));
            }
        } else {
            prev_hook(info);
        }
    }

    fn take_location() -> (String, u32, u32) {
        Self::PANIC_LOCATION.take()
    }
}

impl Drop for SilentPanicGuard {
    fn drop(&mut self) {
        Self::SILENCE_PANICS.with(|c| c.update(|v| v - 1));

        // We do not try to uninstall the hook, since there is
        // no guarantee that our hook is currently installed.
    }
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

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use super::*;
    use core::hint::black_box;
    use std::backtrace::Backtrace;

    #[bench]
    fn catch_unwind(b: &mut test::Bencher) {
        b.iter(|| {
            let f = panic::AssertUnwindSafe(|| black_box(true));
            let r = panic::catch_unwind(black_box(f));
            r.unwrap()
        });
    }

    #[bench]
    fn panic_hook(b: &mut test::Bencher) {
        b.iter(|| {
            let _guard = SilentPanicGuard::new();
        });
    }

    #[bench]
    fn backtrace(b: &mut test::Bencher) {
        b.iter(Backtrace::force_capture);
    }

    #[bench]
    fn backtrace_display(b: &mut test::Bencher) {
        b.iter(|| {
            let bt = Backtrace::force_capture();
            format!("{bt}")
        });
    }
}
