// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{fmt::Debug, marker::PhantomData, num::NonZero, ops::Deref};
use std::sync::{LazyLock, RwLock};

use crate::{
    Effect, Generator, Map, OptionExt as _, Scope, SourceRaw,
    make::{self, CharBuf as _, next_string_impl, next_vec_impl},
    make_char::regex::{
        byte_any, byte_any_non_crlf, byte_any_non_lf, byte_class, char_class, char_non_crlf,
        char_non_lf,
    },
    range::SizeRange,
    read_lock_no_poison, write_lock_no_poison,
};

use regex_syntax::hir::{Class, ClassBytes, ClassUnicode, Hir, HirKind, Look};

const UNABLE_GENERATE_REGEX: &str = "unable to generate value matching the regular expression";

static PARSED_RE: LazyLock<RwLock<Map<String, HirInfo>>> =
    LazyLock::new(|| RwLock::new(Map::default()));

struct HirInfo {
    hir: Hir,
    re: regex::bytes::Regex,
    prefix: PrefixSuffix,
    suffix: PrefixSuffix,
}

fn parse(expr: &str) -> Result<bool, Box<dyn core::error::Error>> {
    let parsed = read_lock_no_poison(&PARSED_RE);
    if let Some(hi) = parsed.get(expr) {
        return Ok(hi.hir.properties().is_utf8());
    }
    drop(parsed);
    let mut parsed = write_lock_no_poison(&PARSED_RE);
    if let Some(hi) = parsed.get(expr) {
        return Ok(hi.hir.properties().is_utf8());
    }
    let hir = regex_syntax::ParserBuilder::new()
        .utf8(false)
        .build()
        .parse(expr)?;
    let re = regex::bytes::Regex::new(expr)
        .expect("internal error: failed to compile regex after successful parse");
    let is_utf8 = hir.properties().is_utf8();
    let (prefix, suffix) = hir_classify_prefix_suffix(&hir);
    parsed.insert(
        expr.to_owned(),
        HirInfo {
            hir,
            re,
            prefix,
            suffix,
        },
    );
    Ok(is_utf8)
}

#[derive(Clone, Copy)]
enum PrefixSuffix {
    Anchor,
    AnchorLF,
    AnchorCRLF,
    NoAnchor,
}

fn hir_classify_prefix_suffix(hir: &Hir) -> (PrefixSuffix, PrefixSuffix) {
    fn hir_find_first_last<'hir>(
        hir: &'hir Hir,
        first: &mut Option<&'hir Hir>,
        last: &mut Option<&'hir Hir>,
    ) {
        match hir.kind() {
            HirKind::Empty | HirKind::Literal(_) | HirKind::Class(_) | HirKind::Look(_) => {
                if first.is_none() {
                    *first = Some(hir);
                }
                *last = Some(hir);
            }
            HirKind::Repetition(repetition) => {
                hir_find_first_last(&repetition.sub, first, last);
            }
            HirKind::Capture(capture) => {
                hir_find_first_last(&capture.sub, first, last);
            }
            HirKind::Concat(hirs) => {
                for hir in hirs {
                    hir_find_first_last(hir, first, last);
                }
            }
            HirKind::Alternation(hirs) => {
                // This is sub-optimal, but simple and probably good enough.
                for hir in hirs {
                    hir_find_first_last(hir, first, last);
                }
            }
        }
    }

    fn hir_classify_first_last(hir: Option<&Hir>) -> PrefixSuffix {
        let hir = hir.expect("internal error: no first/last hir");
        match hir.kind() {
            HirKind::Look(Look::Start | Look::End) => PrefixSuffix::Anchor,
            HirKind::Look(Look::StartLF | Look::EndLF) => PrefixSuffix::AnchorLF,
            HirKind::Look(Look::StartCRLF | Look::EndCRLF) => PrefixSuffix::AnchorCRLF,
            _ => PrefixSuffix::NoAnchor,
        }
    }

    let (mut first, mut last) = (None, None);
    hir_find_first_last(hir, &mut first, &mut last);
    (
        hir_classify_first_last(first),
        hir_classify_first_last(last),
    )
}

enum CharSpecialCases {
    Any,
    AnyNonLF,
    AnyNonCRLF,
}

// `Hir::dot`, Unicode subset.
fn classify_unicode(class: &ClassUnicode) -> Option<CharSpecialCases> {
    match class.ranges() {
        [a] => ((a.start(), a.end()) == ('\x00', '\u{10FFFF}')).then_some(CharSpecialCases::Any),
        [a, b] => ((a.start(), a.end()) == ('\x00', '\x09')
            && (b.start(), b.end()) == ('\x0B', '\u{10FFFF}'))
            .then_some(CharSpecialCases::AnyNonLF),
        [a, b, c] => ((a.start(), a.end()) == ('\x00', '\x09')
            && (b.start(), b.end()) == ('\x0B', '\x0C')
            && (c.start(), c.end()) == ('\x0E', '\u{10FFFF}'))
            .then_some(CharSpecialCases::AnyNonCRLF),
        _ => None,
    }
}

// `Hir::dot`, bytes subset.
fn classify_bytes(class: &ClassBytes) -> Option<CharSpecialCases> {
    match class.ranges() {
        [a] => ((a.start(), a.end()) == (b'\x00', b'\xFF')).then_some(CharSpecialCases::Any),
        [a, b] => ((a.start(), a.end()) == (b'\x00', b'\x09')
            && (b.start(), b.end()) == (b'\x0B', b'\xFF'))
            .then_some(CharSpecialCases::AnyNonLF),
        [a, b, c] => ((a.start(), a.end()) == (b'\x00', b'\x09')
            && (b.start(), b.end()) == (b'\x0B', b'\x0C')
            && (c.start(), c.end()) == (b'\x0E', b'\xFF'))
            .then_some(CharSpecialCases::AnyNonCRLF),
        _ => None,
    }
}

fn next_hir_match(src: &mut SourceRaw, buf: &mut Vec<u8>, hir: &Hir) {
    match hir.kind() {
        HirKind::Empty | HirKind::Look(_) => {
            // Do nothing. In case of `Look`, hope that filtering is enough.
        }
        HirKind::Literal(literal) => {
            buf.extend_from_slice(&literal.0);
        }
        HirKind::Class(class) => match class {
            Class::Unicode(class) => {
                let ch = match classify_unicode(class) {
                    Some(CharSpecialCases::Any) => src.any("", None),
                    Some(CharSpecialCases::AnyNonLF) => src.any_of("", char_non_lf(), None),
                    Some(CharSpecialCases::AnyNonCRLF) => src.any_of("", char_non_crlf(), None),
                    None => src.any_of("", char_class(class), None),
                };
                buf.push_char(ch);
            }
            Class::Bytes(class) => {
                let b = match classify_bytes(class) {
                    Some(CharSpecialCases::Any) => src.any_of("", byte_any(), None),
                    Some(CharSpecialCases::AnyNonLF) => src.any_of("", byte_any_non_lf(), None),
                    Some(CharSpecialCases::AnyNonCRLF) => src.any_of("", byte_any_non_crlf(), None),
                    None => src.any_of("", byte_class(class), None),
                };
                buf.push(b);
            }
        },
        HirKind::Repetition(repetition) => {
            // Could be nice to reuse string generator here in case of char/byte-class repeat.
            let n_steps = if let Some(max) = repetition.max {
                SizeRange::new(repetition.min as usize..=max as usize)
            } else {
                SizeRange::new(repetition.min as usize..)
            };
            let res = src.repeat(
                "",
                Option::<core::iter::Empty<()>>::None,
                n_steps,
                |_n| (),
                |_v, src, _example| {
                    next_hir_match(src, buf, &repetition.sub);
                    Effect::Success
                },
            );
            debug_assert!(res.is_some());
        }
        HirKind::Capture(capture) => {
            next_hir_match(src, buf, &capture.sub);
        }
        HirKind::Concat(hirs) => {
            for hir in hirs {
                let mut src = Scope::new_plain(src, "", "");
                next_hir_match(&mut src, buf, hir);
            }
        }
        HirKind::Alternation(hirs) => {
            let n =
                NonZero::new(hirs.len()).expect("internal error: zero regex alternation variants");
            src.select(
                "",
                None,
                n,
                |_| "",
                |src, _variant, ix| {
                    let hir = &hirs[ix];
                    next_hir_match(src, buf, hir);
                },
            );
        }
    }
}

fn append_pad(
    prefix: bool,
    anchor: PrefixSuffix,
    utf8: bool,
    buf: &mut Vec<u8>,
    src: &mut SourceRaw,
) {
    fn do_append_pad(utf8: bool, buf: &mut Vec<u8>, src: &mut SourceRaw) {
        let pad_size = SizeRange::new(..);
        if utf8 {
            next_string_impl(src, None, buf, make::arbitrary(), pad_size);
        } else {
            next_vec_impl(src, None, buf, byte_any(), pad_size);
        }
    }

    let label = if prefix { "<prefix>" } else { "<suffix>" };
    match anchor {
        PrefixSuffix::Anchor => {
            // No padding possible.
        }
        PrefixSuffix::AnchorLF => {
            src.maybe(label, None, |src| {
                if !prefix {
                    buf.push(b'\n');
                }
                do_append_pad(utf8, buf, src);
                if prefix {
                    buf.push(b'\n');
                }
            });
        }
        PrefixSuffix::AnchorCRLF => {
            src.maybe(label, None, |src| {
                if !prefix {
                    buf.extend_from_slice(b"\r\n");
                }
                do_append_pad(utf8, buf, src);
                if prefix {
                    buf.extend_from_slice(b"\r\n");
                }
            });
        }
        PrefixSuffix::NoAnchor => {
            let mut src = Scope::new_plain(src, label, "");
            do_append_pad(utf8, buf, &mut src);
        }
    }
}

fn next_regex_impl(
    src: &mut SourceRaw,
    _example: Option<&[u8]>, // TODO: examples support
    hi: &HirInfo,
    utf8: bool,
    fullmatch: bool,
) -> Vec<u8> {
    let mut buf = Vec::new();
    if !fullmatch {
        append_pad(true, hi.prefix, utf8, &mut buf, src);
    }
    buf.reserve(hi.hir.properties().minimum_len().unwrap_or_default());
    next_hir_match(src, &mut buf, &hi.hir);
    if !fullmatch {
        append_pad(false, hi.suffix, utf8, &mut buf, src);
    }
    buf
}

fn is_match(full: bool, re: &regex::bytes::Regex, b: &[u8]) -> bool {
    if full {
        re.find(b)
            .is_some_and(|m| (m.start(), m.end()) == (0, b.len()))
    } else {
        re.is_match(b)
    }
}

#[derive(Debug)]
struct RegexString<T> {
    expr: String,
    fullmatch: bool,
    _marker: PhantomData<T>,
}

impl<T> Generator for RegexString<T>
where
    T: From<String> + Deref<Target = str> + Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let parsed = read_lock_no_poison(&PARSED_RE);
        let hi = parsed
            .get(&self.expr)
            .expect("internal error: regular expression not found");
        let v = src.find("<regex>", example, |src, example| {
            let example = example.filter(|e| is_match(self.fullmatch, &hi.re, e.as_bytes()));
            let buf = next_regex_impl(src, example.map(|e| e.as_bytes()), hi, true, self.fullmatch);
            is_match(self.fullmatch, &hi.re, &buf).then(|| {
                String::from_utf8(buf)
                    .expect("internal error: String regex constructed non-UTF-8 sequence")
                    .into()
            })
        });
        v.assume_some_msg(UNABLE_GENERATE_REGEX)
    }
}

/// Create a generator of [`String`] values that match the given regular expression.
///
/// With `fullmatch`, whole string will match the regular expression.
/// Otherwise, whole string or a substring can match.
///
/// # Panics
///
/// `string_matching` panics when the regular expression can't be parsed or when it can match non-UTF-8 sequences.
pub fn string_matching(expr: &str, fullmatch: bool) -> impl Generator<Item = String> + use<> {
    string_slice_matching(expr, fullmatch)
}

/// Create a generator of owned string slice values that match the given regular expression.
///
/// With `fullmatch`, whole slice will match the regular expression.
/// Otherwise, whole slice or a subslice can match.
///
/// Examples of standard owned string slices:
///
/// - [`Box<str>`]
/// - [`Rc<str>`](alloc::rc::Rc)
/// - [`Arc<str>`](alloc::sync::Arc)
/// - [`Cow<'_, str>`](alloc::borrow::Cow)
///
/// # Panics
///
/// `string_slice_matching` panics when the regular expression can't be parsed or when it can match non-UTF-8 sequences.
pub fn string_slice_matching<T>(expr: &str, fullmatch: bool) -> impl Generator<Item = T> + use<T>
where
    T: From<String> + Deref<Target = str> + Debug,
{
    let is_utf8 = parse(expr).expect("invalid regular expression");
    assert!(is_utf8, "regular expression matches non-UTF-8 sequences");
    RegexString {
        expr: expr.to_owned(),
        fullmatch,
        _marker: PhantomData,
    }
}

#[derive(Debug)]
struct RegexBytes<T> {
    expr: String,
    utf8: bool,
    fullmatch: bool,
    _marker: PhantomData<T>,
}

impl<T> Generator for RegexBytes<T>
where
    T: From<Vec<u8>> + Deref<Target = [u8]> + Debug,
{
    type Item = T;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let parsed = read_lock_no_poison(&PARSED_RE);
        let hi = parsed
            .get(&self.expr)
            .expect("internal error: regular expression not found");
        let v = src.find("<regex>", example, |src, example| {
            let example = example.filter(|e| is_match(self.fullmatch, &hi.re, e));
            let buf = next_regex_impl(
                src,
                example.map(Deref::deref),
                hi,
                self.utf8,
                self.fullmatch,
            );
            is_match(self.fullmatch, &hi.re, &buf).then(|| buf.into())
        });
        v.assume_some_msg(UNABLE_GENERATE_REGEX)
    }
}

/// Create a generator of [`Vec<u8>`] values that match the given regular expression.
///
/// With `fullmatch`, whole vector will match the regular expression.
/// Otherwise, whole vector or a subslice can match.
///
/// # Panics
///
/// `bytes_matching` panics when the regular expression can't be parsed.
pub fn bytes_matching(expr: &str, fullmatch: bool) -> impl Generator<Item = Vec<u8>> + use<> {
    byte_slice_matching(expr, fullmatch)
}

/// Create a generator of owned byte slice values that match the given regular expression.
///
/// With `fullmatch`, whole slice will match the regular expression.
/// Otherwise, whole slice or a subslice can match.
///
/// Examples of standard owned byte slices:
///
/// - [`Box<[u8]>`](Box)
/// - [`Rc<[u8]>`](alloc::rc::Rc)
/// - [`Arc<[u8]>`](alloc::sync::Arc)
/// - [`Cow<'_, [u8]>`](alloc::borrow::Cow)
///
/// # Panics
///
/// `byte_slice_matching` panics when the regular expression can't be parsed.
pub fn byte_slice_matching<T>(expr: &str, fullmatch: bool) -> impl Generator<Item = T> + use<T>
where
    T: From<Vec<u8>> + Deref<Target = [u8]> + Debug,
{
    let utf8 = parse(expr).expect("invalid regular expression");
    RegexBytes {
        expr: expr.to_owned(),
        utf8,
        fullmatch,
        _marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use crate::{check, make, tests::any_assert_valid_tape, vdbg};

    fn check_regex_consistent(pattern: &'static str) {
        let re = regex::Regex::new(pattern).unwrap();
        check(|src| {
            vdbg!(src, pattern);
            let fullmatch = src.any("fullmatch");
            let g = make::string_matching(pattern, fullmatch);
            let s = any_assert_valid_tape(src, "s", g);
            assert!(
                re.find(&s)
                    .is_some_and(|m| { !fullmatch || (m.start(), m.end()) == (0, s.len()) })
            );
        });

        let re = regex::bytes::Regex::new(pattern).unwrap();
        check(|src| {
            vdbg!(src, pattern);
            let fullmatch = src.any("fullmatch");
            let g = make::bytes_matching(pattern, fullmatch);
            let s = any_assert_valid_tape(src, "s", g);
            assert!(
                re.find(&s)
                    .is_some_and(|m| { !fullmatch || (m.start(), m.end()) == (0, s.len()) })
            );
        });
    }

    // Only `bullet_core_2` and `bullet_core_3` contain non-NFC literals,
    // but for simplicity we allow them for everything.
    macro_rules! consistent {
        ($name:ident, $pattern:expr) => {
            #[test]
            #[expect(clippy::allow_attributes)]
            #[allow(clippy::unicode_not_nfc)]
            fn $name() {
                check_regex_consistent($pattern);
            }
        };
    }

    consistent!(empty, "");

    include!("tests_crates_regex.rs");
}
