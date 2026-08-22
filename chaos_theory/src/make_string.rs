// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::{borrow::Cow, boxed::Box, ffi::CString, rc::Rc, string::String, sync::Arc, vec::Vec};
use core::{
    ffi::CStr,
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, RangeBounds},
    str::{Chars, Utf8Chunks},
};
#[cfg(all(feature = "std", target_os = "hermit"))]
use std::os::hermit::ffi::{OsStrExt as _, OsStringExt as _};
#[cfg(all(feature = "std", target_os = "solid_asp3"))]
use std::os::solid::ffi::{OsStrExt as _, OsStringExt as _};
#[cfg(all(feature = "std", unix))]
use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};
#[cfg(all(feature = "std", target_os = "wasi"))]
use std::os::wasi::ffi::{OsStrExt as _, OsStringExt as _};
#[cfg(all(feature = "std", windows))]
use std::os::windows::ffi::{OsStrExt as _, OsStringExt as _};
#[cfg(all(feature = "std", target_os = "xous"))]
use std::os::xous::ffi::{OsStrExt as _, OsStringExt as _};
#[cfg(feature = "std")]
use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    sync::LazyLock,
};

use crate::{
    Arbitrary, Effect, Generator, MaybeOwned, SourceEx, make_collection::next_vec_impl,
    math::percent, range::SizeRange,
};

const STRING_SPECIAL_PROB: f64 = percent(10);
const REPLACEMENT_CHAR: char = '\u{FFFD}';

impl Arbitrary for CString {
    fn arbitrary() -> impl Generator<Item = Self> {
        cstring(u8::arbitrary())
    }
}

impl Arbitrary for Box<CStr> {
    fn arbitrary() -> impl Generator<Item = Self> {
        cstring_slice(u8::arbitrary())
    }
}

impl Arbitrary for Rc<CStr> {
    fn arbitrary() -> impl Generator<Item = Self> {
        cstring_slice(u8::arbitrary())
    }
}

impl Arbitrary for Arc<CStr> {
    fn arbitrary() -> impl Generator<Item = Self> {
        cstring_slice(u8::arbitrary())
    }
}

impl Arbitrary for Cow<'_, CStr> {
    fn arbitrary() -> impl Generator<Item = Self> {
        cstring_slice(u8::arbitrary())
    }
}

#[cfg(feature = "std")]
impl Arbitrary for OsString {
    fn arbitrary() -> impl Generator<Item = Self> {
        os_string_arbitrary()
    }
}

#[cfg(feature = "std")]
impl Arbitrary for Box<OsStr> {
    fn arbitrary() -> impl Generator<Item = Self> {
        os_string_arbitrary()
    }
}

#[cfg(feature = "std")]
impl Arbitrary for Rc<OsStr> {
    fn arbitrary() -> impl Generator<Item = Self> {
        os_string_arbitrary()
    }
}

#[cfg(feature = "std")]
impl Arbitrary for Arc<OsStr> {
    fn arbitrary() -> impl Generator<Item = Self> {
        os_string_arbitrary()
    }
}

#[cfg(feature = "std")]
impl Arbitrary for Cow<'_, OsStr> {
    fn arbitrary() -> impl Generator<Item = Self> {
        os_string_arbitrary()
    }
}

#[cfg(feature = "std")]
impl Arbitrary for PathBuf {
    fn arbitrary() -> impl Generator<Item = Self> {
        path_buf_arbitrary()
    }
}

#[cfg(feature = "std")]
impl Arbitrary for Box<Path> {
    fn arbitrary() -> impl Generator<Item = Self> {
        path_arbitrary()
    }
}

#[cfg(feature = "std")]
impl Arbitrary for Rc<Path> {
    fn arbitrary() -> impl Generator<Item = Self> {
        path_arbitrary()
    }
}

#[cfg(feature = "std")]
impl Arbitrary for Arc<Path> {
    fn arbitrary() -> impl Generator<Item = Self> {
        path_arbitrary()
    }
}

#[cfg(feature = "std")]
impl Arbitrary for Cow<'_, Path> {
    fn arbitrary() -> impl Generator<Item = Self> {
        path_arbitrary()
    }
}

impl Arbitrary for String {
    fn arbitrary() -> impl Generator<Item = Self> {
        string(char::arbitrary())
    }
}

impl Arbitrary for Box<str> {
    fn arbitrary() -> impl Generator<Item = Self> {
        string_slice(char::arbitrary())
    }
}

impl Arbitrary for Rc<str> {
    fn arbitrary() -> impl Generator<Item = Self> {
        string_slice(char::arbitrary())
    }
}

impl Arbitrary for Arc<str> {
    fn arbitrary() -> impl Generator<Item = Self> {
        string_slice(char::arbitrary())
    }
}

impl Arbitrary for Cow<'_, str> {
    fn arbitrary() -> impl Generator<Item = Self> {
        string_slice(char::arbitrary())
    }
}

// List based on https://github.com/minimaxir/big-list-of-naughty-strings/blob/master/blns.txt
const STRING_SPECIAL: &[&str] = &[
    // Strings which may be used elsewhere in code
    "undefined",
    "undef",
    "null",
    "NULL",
    "(null)",
    "nil",
    "NIL",
    "true",
    "false",
    "True",
    "False",
    "TRUE",
    "FALSE",
    "None",
    "hasOwnProperty",
    "then",
    "constructor",
    "__class__",
    // Strings which can be interpreted as numeric
    "0",
    "1E100",
    "0/0",
    "1/0",
    "0..0",
    "NaN",
    "Infinity",
    "-Infinity",
    "INF",
    "1#INF",
    "0xffffffffffffffff",
    "123456789012345678901234567890123456789",
    "1,000.00",
    "0755",
    // Unwanted interpolation
    "$HOME",
    "%d",
    "%s%s%s%s%s",
    "{0}",
    "%*.*s",
    "%@",
    // Path shapes which make lexical operations useful
    "a/b.tar.gz",
    "a/.b.toml",
    "./a",
    "../a",
    "a/b/",
    "a\0b",
    // Rooted paths
    "/a/b.tar.gz",
    "//a/b/c.tar.gz",
    // Windows path prefixes and special names
    r"\a\b.txt",
    r"C:a\b.txt",
    r"C:\a\b.txt",
    r"\\a\b\c.txt",
    r"\\?\a\b.txt",
    r"\\?\C:\a\b.txt",
    r"\\?\UNC\a\b\c.txt",
    r"\\.\a\b.txt",
    r"a\b.txt:stream",
    r"a\b. ",
    "CON",
    "PRN",
    "AUX",
    "CLOCK$",
    "NUL",
    "NUL.txt",
];

// Index of the first path-specific entry (`"a/b.tar.gz"`).
const PATH_SPECIAL_START: usize = 38;
const PATH_SPECIAL: &[&str] = STRING_SPECIAL.split_at(PATH_SPECIAL_START).1;

#[cfg(feature = "std")]
static PATH_SPECIAL_VALUES: LazyLock<Vec<PathBuf>> =
    LazyLock::new(|| PATH_SPECIAL.iter().map(PathBuf::from).collect());

#[derive(Debug)]
struct CString_<G, T> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<T>,
}

impl<G, T> Generator for CString_<G, T>
where
    G: Generator<Item = u8>,
    T: From<CString> + Deref<Target = CStr> + Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let mut bytes = Vec::new();
        let example = example.map(|s| s.to_bytes()).or_else(|| {
            src.as_mut()
                .choose_seed(STRING_SPECIAL_PROB, STRING_SPECIAL)
                .copied()
                .map(str::as_bytes)
        });
        next_vec_impl(src, example, &mut bytes, &self.elem, self.size);
        CString::new(bytes)
            .expect("internal error: generated NUL byte")
            .into()
    }
}

#[cfg(all(
    feature = "std",
    any(
        unix,
        target_os = "hermit",
        target_os = "solid_asp3",
        target_os = "wasi",
        target_os = "xous"
    )
))]
#[derive(Debug)]
struct OsStringBytes_<G, T> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<T>,
}

#[cfg(all(
    feature = "std",
    any(
        unix,
        target_os = "hermit",
        target_os = "solid_asp3",
        target_os = "wasi",
        target_os = "xous"
    )
))]
impl<G, T> Generator for OsStringBytes_<G, T>
where
    G: Generator<Item = u8>,
    T: From<OsString> + AsRef<OsStr> + Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let mut bytes = Vec::new();
        let example = example.map(|s| s.as_ref().as_bytes()).or_else(|| {
            src.as_mut()
                .choose_seed(STRING_SPECIAL_PROB, STRING_SPECIAL)
                .copied()
                .map(str::as_bytes)
        });
        next_vec_impl(src, example, &mut bytes, &self.elem, self.size);
        OsString::from_vec(bytes).into()
    }
}

#[cfg(all(
    feature = "std",
    any(
        unix,
        target_os = "hermit",
        target_os = "solid_asp3",
        target_os = "wasi",
        target_os = "xous"
    )
))]
fn os_string_arbitrary<T>() -> impl Generator<Item = T>
where
    T: From<OsString> + AsRef<OsStr> + Debug,
{
    OsStringBytes_ {
        elem: u8::arbitrary(),
        size: SizeRange::new(..),
        _marker: PhantomData,
    }
}

#[cfg(all(feature = "std", windows))]
#[derive(Debug)]
struct OsStringWide_<G, T> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<T>,
}

#[cfg(all(feature = "std", windows))]
impl<G, T> Generator for OsStringWide_<G, T>
where
    G: Generator<Item = u16>,
    T: From<OsString> + AsRef<OsStr> + Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let mut wide = Vec::new();
        let example = example
            .map(|s| s.as_ref().encode_wide().collect::<Vec<_>>())
            .or_else(|| {
                src.as_mut()
                    .choose_seed(STRING_SPECIAL_PROB, STRING_SPECIAL)
                    .copied()
                    .map(|s| s.encode_utf16().collect())
            });
        next_vec_impl(src, example.as_deref(), &mut wide, &self.elem, self.size);
        OsString::from_wide(&wide).into()
    }
}

#[cfg(all(feature = "std", windows))]
fn os_string_arbitrary<T>() -> impl Generator<Item = T>
where
    T: From<OsString> + AsRef<OsStr> + Debug,
{
    OsStringWide_ {
        elem: u16::arbitrary(),
        size: SizeRange::new(..),
        _marker: PhantomData,
    }
}

#[cfg(all(
    feature = "std",
    not(any(
        unix,
        windows,
        target_os = "hermit",
        target_os = "solid_asp3",
        target_os = "wasi",
        target_os = "xous"
    ))
))]
fn os_string_arbitrary<T>() -> impl Generator<Item = T>
where
    T: From<OsString> + AsRef<OsStr> + Debug,
{
    String::arbitrary().map_reversible(
        |s| OsString::from(s).into(),
        |s: &T| {
            s.as_ref()
                .to_str()
                .map(|s| MaybeOwned::Owned(String::from(s)))
        },
    )
}

#[cfg(feature = "std")]
fn path_buf_arbitrary() -> impl Generator<Item = PathBuf> {
    os_string_arbitrary().seeded(PATH_SPECIAL_VALUES.as_slice(), true)
}

#[cfg(feature = "std")]
fn path_arbitrary<T>() -> impl Generator<Item = T>
where
    T: From<PathBuf> + Deref<Target = Path> + Debug,
{
    path_buf_arbitrary()
        .map_reversible(Into::into, |p: &T| Some(MaybeOwned::Owned(p.to_path_buf())))
}

impl<G, T> Generator for String_<G, T>
where
    G: Generator<Item = char>,
    T: From<String> + Deref<Target = str> + Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let mut s = String::new();
        next_string_impl(
            src,
            example.map(|e| e.as_bytes()),
            &mut s,
            &self.elem,
            self.size,
        );
        s.into()
    }
}

pub(crate) trait CharBuf: core::fmt::Debug {
    fn reserve(&mut self, size: usize);
    fn push_char(&mut self, ch: char);
}

impl CharBuf for String {
    fn reserve(&mut self, size: usize) {
        self.reserve(size);
    }

    fn push_char(&mut self, ch: char) {
        self.push(ch);
    }
}

impl CharBuf for Vec<u8> {
    fn reserve(&mut self, size: usize) {
        self.reserve(size);
    }

    fn push_char(&mut self, ch: char) {
        match ch.len_utf8() {
            1 => self.push(ch as u8),
            _ => self.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes()),
        }
    }
}

struct CharIterator<'a> {
    chunks: Utf8Chunks<'a>,
    cur_chars: Option<Chars<'a>>,
    cur_invalid: bool,
    len: usize,
}

impl<'a> CharIterator<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        let mut len = 0;
        for chunk in bytes.utf8_chunks() {
            for _ in chunk.valid().chars() {
                len += 1;
            }
            len += usize::from(!chunk.invalid().is_empty());
        }
        let mut s = Self {
            chunks: bytes.utf8_chunks(),
            cur_chars: None,
            cur_invalid: false,
            len,
        };
        s.next_chunk();
        s
    }

    fn next_chunk(&mut self) {
        let cur_chunk = self.chunks.next();
        self.cur_chars = cur_chunk.as_ref().map(|c| c.valid().chars());
        self.cur_invalid = cur_chunk.is_some_and(|c| !c.invalid().is_empty());
    }
}

impl Iterator for CharIterator<'_> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(cur_chars) = &mut self.cur_chars {
            let ch = cur_chars.next();
            if ch.is_some() {
                return ch;
            }
            let ch = self.cur_invalid.then_some(REPLACEMENT_CHAR);
            self.next_chunk();
            if ch.is_some() {
                return ch;
            }
        }
        None
    }
}

impl ExactSizeIterator for CharIterator<'_> {
    fn len(&self) -> usize {
        self.len
    }
}

// String generation is not implemented with `collect` because that would require defining
// a new trait that would need to be implemented for both `IntoIterator<IntoIter: ExactSizeIterator>`
// types (like most collections) and String, which is impossible due to the coherence rules.
pub(crate) fn next_string_impl(
    src: &mut SourceEx,
    mut example: Option<&[u8]>,
    s: &mut impl CharBuf,
    elem: impl Generator<Item = char>,
    size: SizeRange,
) {
    if example.is_none() {
        example = src
            .as_mut()
            .choose_seed(STRING_SPECIAL_PROB, STRING_SPECIAL)
            .copied()
            .map(str::as_bytes);
    }
    let example_char_seq = example.map(CharIterator::new);
    let res = src.repeat(
        "<string>",
        example_char_seq,
        size,
        |n| {
            // Wasteful, but guaranteed not to reallocate.
            s.reserve(n * 4);
            s
        },
        |s, src, example| {
            let elem = elem.next(src, example.as_ref());
            s.push_char(elem);
            Effect::Success
        },
    );
    debug_assert!(res.is_some());
}

#[derive(Debug)]
struct String_<G, T> {
    elem: G,
    size: SizeRange,
    _marker: PhantomData<T>,
}

/// Create a [`CString`] generator.
///
/// NUL bytes produced by `elem` are mapped to ASCII spaces.
pub fn cstring(elem: impl Generator<Item = u8>) -> impl Generator<Item = CString> {
    cstring_slice(elem)
}

/// Create a [`CString`] generator with the specified content size in bytes, excluding the terminating NUL.
///
/// NUL bytes produced by `elem` are mapped to ASCII spaces.
pub fn cstring_n(
    elem: impl Generator<Item = u8>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = CString> {
    cstring_slice_n(elem, size)
}

/// Create an owned C string slice generator.
///
/// NUL bytes produced by `elem` are mapped to ASCII spaces.
///
/// Examples of standard owned C string slices:
///
/// - [`Box<CStr>`]
/// - [`Rc<CStr>`]
/// - [`Arc<CStr>`]
/// - [`Cow<'_, CStr>`](alloc::borrow::Cow)
pub fn cstring_slice<T>(elem: impl Generator<Item = u8>) -> impl Generator<Item = T>
where
    T: From<CString> + Deref<Target = CStr> + Debug,
{
    cstring_slice_n(elem, ..)
}

/// Create an owned C string slice generator with the specified content size in bytes,
/// excluding the terminating NUL.
///
/// NUL bytes produced by `elem` are mapped to ASCII spaces.
///
/// Examples of standard owned C string slices:
///
/// - [`Box<CStr>`]
/// - [`Rc<CStr>`]
/// - [`Arc<CStr>`]
/// - [`Cow<'_, CStr>`](alloc::borrow::Cow)
pub fn cstring_slice_n<T>(
    elem: impl Generator<Item = u8>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = T>
where
    T: From<CString> + Deref<Target = CStr> + Debug,
{
    let elem = elem.map_reversible(
        |b| if b == b'\0' { b' ' } else { b },
        |b| Some(MaybeOwned::Borrowed(b)),
    );
    CString_ {
        elem,
        size: SizeRange::new(size),
        _marker: PhantomData,
    }
}

/// Create a [`String`] generator.
pub fn string(elem: impl Generator<Item = char>) -> impl Generator<Item = String> {
    string_n(elem, ..)
}

// TODO: limit the length of the string (in bytes) - will need Effect::Done

/// Create a [`String`] generator with the specified size (in characters).
pub fn string_n(
    elem: impl Generator<Item = char>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = String> {
    string_slice_n(elem, size)
}

/// Create an owned string slice generator.
///
/// Examples of standard owned string slices:
///
/// - [`Box<str>`]
/// - [`Rc<str>`]
/// - [`Arc<str>`]
/// - [`Cow<'_, str>`](alloc::borrow::Cow)
pub fn string_slice<T>(elem: impl Generator<Item = char>) -> impl Generator<Item = T>
where
    T: From<String> + Deref<Target = str> + Debug,
{
    string_slice_n(elem, ..)
}

/// Create an owned string slice generator with the specified size (in characters).
///
/// Examples of standard owned string slices:
///
/// - [`Box<str>`]
/// - [`Rc<str>`]
/// - [`Arc<str>`]
/// - [`Cow<'_, str>`](alloc::borrow::Cow)
pub fn string_slice_n<T>(
    elem: impl Generator<Item = char>,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = T>
where
    T: From<String> + Deref<Target = str> + Debug,
{
    let size = SizeRange::new(size);
    String_ {
        elem,
        size,
        _marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use alloc::{borrow::Cow, boxed::Box, rc::Rc, sync::Arc};

    use crate::{
        Arbitrary as _, Generator as _, check, make,
        tests::{print_debug_examples, prop_smoke},
    };

    use super::{CStr, CString, CharIterator};
    #[cfg(feature = "std")]
    use std::{
        ffi::{OsStr, OsString},
        path::{Path, PathBuf},
    };

    #[test]
    fn char_iterator() {
        check(|src| {
            let bytes: Vec<u8> = src.any("bytes");
            let iter = CharIterator::new(&bytes);
            let char_len = iter.len();
            let s1: Vec<char> = iter.collect();
            let s2: Vec<char> = String::from_utf8_lossy(&bytes).chars().collect();
            assert_eq!(s1.len(), char_len);
            assert_eq!(s1, s2);
        });
    }

    #[test]
    fn c_string_smoke() {
        check(|src| {
            prop_smoke(src, "CString", CString::arbitrary());
            prop_smoke(src, "Box<CStr>", Box::<CStr>::arbitrary());
            prop_smoke(src, "Rc<CStr>", Rc::<CStr>::arbitrary());
            prop_smoke(src, "Arc<CStr>", Arc::<CStr>::arbitrary());
            prop_smoke(src, "Cow<CStr>", Cow::<CStr>::arbitrary());
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn os_string_smoke() {
        check(|src| {
            prop_smoke(src, "OSString", OsString::arbitrary());
            prop_smoke(src, "Box<OsStr>", Box::<OsStr>::arbitrary());
            prop_smoke(src, "Rc<OsStr>", Rc::<OsStr>::arbitrary());
            prop_smoke(src, "Arc<OsStr>", Arc::<OsStr>::arbitrary());
            prop_smoke(src, "Cow<OsStr>", Cow::<OsStr>::arbitrary());
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn path_smoke() {
        check(|src| {
            prop_smoke(src, "PathBuf", PathBuf::arbitrary());
            prop_smoke(src, "Box<Path>", Box::<Path>::arbitrary());
            prop_smoke(src, "Rc<Path>", Rc::<Path>::arbitrary());
            prop_smoke(src, "Arc<Path>", Arc::<Path>::arbitrary());
            prop_smoke(src, "Cow<Path>", Cow::<Path>::arbitrary());
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn path_examples() {
        print_debug_examples(&make::arbitrary::<PathBuf>(), None, |a, b| {
            (a.as_os_str().len(), a).cmp(&(b.as_os_str().len(), b))
        });
    }

    #[test]
    fn string_smoke() {
        check(|src| {
            prop_smoke(src, "string", String::arbitrary());
        });
    }

    #[test]
    fn string_gen_example() {
        check(|src| {
            let example = src.any_of(
                "example",
                make::arbitrary::<char>()
                    // Without limiting the size, the example can use all the budget and the string will get back short.
                    .collect_n::<Vec<_>>(..64),
            );
            let example = String::from_iter(example);
            let s = src.as_ex().any("s", Some(&example));
            assert_eq!(s, example);
        });
    }

    #[test]
    fn string_examples() {
        print_debug_examples(&make::arbitrary::<String>(), None, |a, b| {
            (a.len(), a).cmp(&(b.len(), b))
        });
    }

    #[test]
    fn string_like_examples() {
        print_debug_examples(
            &make::arbitrary::<String>(),
            Some(&"hello".into()),
            |a, b| (a.len(), a).cmp(&(b.len(), b)),
        );
    }

    #[test]
    fn string_seeded_examples() {
        print_debug_examples(
            &make::arbitrary::<String>().seeded(&["hello".into()], true),
            None,
            |a, b| (a.len(), a).cmp(&(b.len(), b)),
        );

        print_debug_examples(
            &make::string(make::arbitrary::<char>().seeded(&['Z'], true))
                .seeded(&["hey".into()], true),
            None,
            |a, b| (a.len(), a).cmp(&(b.len(), b)),
        );
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use crate::{Arbitrary as _, tests::bench_gen_next};

    #[bench]
    fn gen_string(b: &mut test::Bencher) {
        let g = String::arbitrary();
        bench_gen_next(b, &g);
    }
}
