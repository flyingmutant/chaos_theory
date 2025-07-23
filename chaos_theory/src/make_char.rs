// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::cmp::Ordering;

use crate::{
    Arbitrary, Generator, SourceRaw, Tweak,
    make::{BYTE_SPECIAL, BYTE_SPECIAL_PROB},
    math::percent,
};

impl Arbitrary for char {
    fn arbitrary() -> impl Generator<Item = Self> {
        Char {
            cr_to_space: false,
            lf_to_space: false,
        }
    }
}

const CHAR_SPECIAL_PROB: f64 = percent(75);

const CHAR_SPECIAL: &[char] = &[
    // A-Z
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', //
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', //
    // a-z
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', //
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', //
    // 0-9
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', //
    // A bit of Latin-1
    'À', 'æ', '×', '÷', '¿', //
    // Top row symbols
    '~', '!', '@', '#', '$', '%', '^', '&', '*', '-', '_', '=', '+', //
    // Punctuation
    '.', ',', ';', ':', '?', //
    // Whitespace
    ' ', '\t', '\r', '\n', //
    // Separators
    '/', '\\', '|', //
    // Brackets
    '(', ')', '[', ']', '{', '}', '<', '>', //
    // Quotes
    '\'', '"', '`', //
    // NUL, VT, ESC, DEL
    '\x00', '\x0B', '\x1B', '\x7F', //
    // Replacement character, BOM, RTL override
    '\u{FFFD}', '\u{FEFF}', '\u{202E}', // cspell:disable-line
    // In UTF-8, Ⱥ increases in length from 2 to 3 bytes when lowercased
    'Ⱥ', //
    // In Shift JIS, ¥ has code 0x5C, normally used for backslash escape
    '¥', //
    // No non-Unicode encoding has both ¥ and Ѩ
    'Ѩ', //
    // Non-BMP character
    '🔥', //
];

#[rustfmt::skip]
#[expect(dead_code)]
#[expect(clippy::redundant_static_lifetimes)]
#[expect(clippy::unicode_not_nfc)]
mod cat {
    include!("../gen/ucd_general_category.rs");
}

#[rustfmt::skip]
#[expect(clippy::redundant_static_lifetimes)]
#[expect(clippy::unicode_not_nfc)]
mod cat_enum {
    include!("../gen/ucd_general_category_enum.rs");
}

// All general categories excluding surrogates, sorted by taste.
const SORTED_CATEGORIES: &[(&str, &[(char, char)])] = &[
    // 29 Unicode general categories (all except the surrogates)
    ("Uppercase_Letter", cat::UPPERCASE_LETTER), // Lu
    ("Lowercase_Letter", cat::LOWERCASE_LETTER), // Ll
    ("Titlecase_Letter", cat::TITLECASE_LETTER), // Lt
    ("Modifier_Letter", cat::MODIFIER_LETTER),   // Lm
    ("Other_Letter", cat::OTHER_LETTER),         // Lo
    ("Decimal_Number", cat::DECIMAL_NUMBER),     // Nd
    ("Letter_Number", cat::LETTER_NUMBER),       // Nl
    ("Other_Number", cat::OTHER_NUMBER),         // No
    ("Connector_Punctuation", cat::CONNECTOR_PUNCTUATION), // Pc
    ("Dash_Punctuation", cat::DASH_PUNCTUATION), // Pd
    ("Open_Punctuation", cat::OPEN_PUNCTUATION), // Ps
    ("Close_Punctuation", cat::CLOSE_PUNCTUATION), // Pe
    ("Initial_Punctuation", cat::INITIAL_PUNCTUATION), // Pi
    ("Final_Punctuation", cat::FINAL_PUNCTUATION), // Pf
    ("Other_Punctuation", cat::OTHER_PUNCTUATION), // Po
    ("Math_Symbol", cat::MATH_SYMBOL),           // Sm
    ("Currency_Symbol", cat::CURRENCY_SYMBOL),   // Sc
    ("Modifier_Symbol", cat::MODIFIER_SYMBOL),   // Sk
    ("Other_Symbol", cat::OTHER_SYMBOL),         // So
    ("Nonspacing_Mark", cat::NONSPACING_MARK),   // Mn
    ("Spacing_Mark", cat::SPACING_MARK),         // Mc
    ("Enclosing_Mark", cat::ENCLOSING_MARK),     // Me
    ("Space_Separator", cat::SPACE_SEPARATOR),   // Zs
    ("Line_Separator", cat::LINE_SEPARATOR),     // Zl
    ("Paragraph_Separator", cat::PARAGRAPH_SEPARATOR), // Zp
    ("Control", cat::CONTROL),                   // Cc
    ("Format", cat::FORMAT),                     // Cf
    ("Private_Use", cat::PRIVATE_USE),           // Co
    ("Unassigned", cat::UNASSIGNED),             // Cn
];

fn char_category_name(c: char) -> &'static str {
    let cat_enum_ix = cat_enum::GENERAL_CATEGORY
        .binary_search_by(|(a, b, _cat)| {
            if *b < c {
                Ordering::Less
            } else if *a > c {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .expect("internal error: can't find character unicode category");
    let cat_name_ix = cat_enum::GENERAL_CATEGORY[cat_enum_ix].2;
    cat_enum::GENERAL_CATEGORY_ENUM[cat_name_ix as usize]
}

fn category_index_by_name(cat: &'static str) -> usize {
    // Yep, linear search, quite slow.
    SORTED_CATEGORIES
        .iter()
        .position(|(name, _cat)| *name == cat)
        .expect("internal error: can't find unicode category by name")
}

fn char_category_index(c: char) -> usize {
    let cat = char_category_name(c);
    category_index_by_name(cat)
}

fn char_range_index(c: char, cat: &[(char, char)]) -> usize {
    cat.binary_search_by(|(a, b)| {
        if *b < c {
            Ordering::Less
        } else if *a > c {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    })
    .expect("internal error: can't find character in it's category")
}

impl Generator for Char {
    type Item = char;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let mut example = example.copied();
        if example.is_none() {
            example = src
                .as_mut()
                .choose_seed(CHAR_SPECIAL_PROB, CHAR_SPECIAL)
                .copied();
        }
        let example_cat_ix = example.map(char_category_index);
        let cat_ix =
            src.as_mut()
                .choose_index(SORTED_CATEGORIES.len(), example_cat_ix, Tweak::CharCategory);
        let cat = SORTED_CATEGORIES[cat_ix].1;
        let example_range_ix = example.map(|c| char_range_index(c, cat));
        let range_ix = src
            .as_mut()
            .choose_index(cat.len(), example_range_ix, Tweak::CharRange);
        let range = &cat[range_ix];
        let example_ix = example.map(|c| c as usize - range.0 as usize);
        let c_ix = src.as_mut().choose_index(
            range.1 as usize - range.0 as usize + 1,
            example_ix,
            Tweak::CharIndex,
        );
        let mut ch = char::from_u32(range.0 as u32 + c_ix as u32)
            .expect("internal error: generated invalid char");
        if (self.cr_to_space && ch == '\r') || (self.lf_to_space && ch == '\n') {
            ch = ' ';
        }
        ch
    }
}

#[derive(Debug)]
struct Char {
    lf_to_space: bool,
    cr_to_space: bool,
}

#[derive(Debug)]
// We could have used int_in_range::<u8>(..=0x7f),
// but that is based on `choose_value`, and we prefer `choose_index` here.
struct Ascii {}

impl Generator for Ascii {
    type Item = u8;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        const ASCII_MASK: u8 = 0x7f;
        let mut example = example.copied();
        if example.is_none() {
            example = src
                .as_mut()
                .choose_seed(BYTE_SPECIAL_PROB, BYTE_SPECIAL)
                .copied();
        }
        let example = example.map(|e| usize::from(e & ASCII_MASK));
        let b = src
            .as_mut()
            .choose_index(usize::from(ASCII_MASK + 1), example, Tweak::None) as u8;
        debug_assert!(char::from(b).is_ascii());
        b
    }
}

/// Create a generator of ASCII [`char`] values.
pub fn char_ascii() -> impl Generator<Item = char> {
    byte_ascii().map_into_try_from()
}

/// Create a generator of ASCII [`u8`] values.
pub fn byte_ascii() -> impl Generator<Item = u8> {
    Ascii {}
}

#[cfg(feature = "regex")]
pub(crate) mod regex {
    use super::Char;
    use crate::{
        Generator, Tweak,
        make::{BYTE_SPECIAL, BYTE_SPECIAL_PROB, from_fn_assume},
    };
    use core::cmp::Ordering;
    use regex_syntax::hir::{ClassBytes, ClassBytesRange, ClassUnicode, ClassUnicodeRange};

    fn char_range_index_unicode(c: char, ranges: &[ClassUnicodeRange]) -> Option<usize> {
        ranges
            .binary_search_by(|r| {
                if r.end() < c {
                    Ordering::Less
                } else if r.start() > c {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .ok()
    }

    fn char_range_index_bytes(b: u8, ranges: &[ClassBytesRange]) -> Option<usize> {
        ranges
            .binary_search_by(|r| {
                if r.end() < b {
                    Ordering::Less
                } else if r.start() > b {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .ok()
    }

    fn next_char_unicode(
        src: &mut crate::SourceRaw,
        example: Option<&char>,
        class: &ClassUnicode,
    ) -> Option<char> {
        let example = example.copied();
        let ranges = class.ranges();
        let example_range_ix = example.and_then(|c| char_range_index_unicode(c, ranges));
        let range_ix = src
            .as_mut()
            .choose_index(ranges.len(), example_range_ix, Tweak::CharRange);
        let range = &ranges[range_ix];
        let example_ix = example.map(|c| c as usize - range.start() as usize);
        let c_ix = src.as_mut().choose_index(
            range.end() as usize - range.start() as usize + 1,
            example_ix,
            Tweak::CharIndex,
        );
        char::from_u32(range.start() as u32 + c_ix as u32)
    }

    pub(crate) fn char_class(class: &ClassUnicode) -> impl Generator<Item = char> {
        // This is kind of slow. In theory, we could pre-process Hir to eliminate surrogate ranges instead.
        from_fn_assume(|src, example| next_char_unicode(src, example, class))
    }

    #[derive(Debug)]
    struct CharClassBytes<'a> {
        class: Option<&'a ClassBytes>,
        cr_to_space: bool,
        lf_to_space: bool,
    }

    impl Generator for CharClassBytes<'_> {
        type Item = u8;

        fn next(&self, src: &mut crate::SourceRaw, example: Option<&Self::Item>) -> Self::Item {
            let mut example = example.copied();
            if example.is_none() {
                example = src
                    .as_mut()
                    .choose_seed(BYTE_SPECIAL_PROB, BYTE_SPECIAL)
                    .copied();
            }
            let whole_range = [ClassBytesRange::new(u8::MIN, u8::MAX)];
            let ranges = self.class.map_or(whole_range.as_slice(), |c| c.ranges());
            let example_range_ix = example.and_then(|c| char_range_index_bytes(c, ranges));
            let range_ix =
                src.as_mut()
                    .choose_index(ranges.len(), example_range_ix, Tweak::CharRange);
            let range = &ranges[range_ix];
            let example_ix = example.map(|c| c as usize - range.start() as usize);
            let c_ix = src.as_mut().choose_index(
                range.end() as usize - range.start() as usize + 1,
                example_ix,
                Tweak::CharIndex,
            );
            let mut ch = range.start() + c_ix as u8;
            if (self.cr_to_space && ch == b'\r') || (self.lf_to_space && ch == b'\n') {
                ch = b' ';
            }
            ch
        }
    }

    pub(crate) fn char_non_lf() -> impl Generator<Item = char> {
        Char {
            cr_to_space: false,
            lf_to_space: true,
        }
    }

    pub(crate) fn char_non_crlf() -> impl Generator<Item = char> {
        Char {
            cr_to_space: true,
            lf_to_space: true,
        }
    }

    pub(crate) fn byte_any() -> impl Generator<Item = u8> {
        CharClassBytes {
            class: None,
            cr_to_space: false,
            lf_to_space: false,
        }
    }

    pub(crate) fn byte_any_non_lf() -> impl Generator<Item = u8> {
        CharClassBytes {
            class: None,
            cr_to_space: false,
            lf_to_space: true,
        }
    }

    pub(crate) fn byte_any_non_crlf() -> impl Generator<Item = u8> {
        CharClassBytes {
            class: None,
            cr_to_space: true,
            lf_to_space: true,
        }
    }

    pub(crate) fn byte_class(class: &ClassBytes) -> impl Generator<Item = u8> {
        CharClassBytes {
            class: Some(class),
            cr_to_space: false,
            lf_to_space: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        check, make,
        tests::{print_debug_examples, prop_smoke},
    };
    use core::ops::RangeInclusive;

    #[test]
    fn char_smoke() {
        check(|src| {
            prop_smoke(src, "char", char::arbitrary());
            prop_smoke(src, "char_ascii", char_ascii());
            prop_smoke(src, "byte_ascii", byte_ascii());
        });
    }

    #[test]
    fn char_gen_example() {
        check(|src| {
            const SURROGATE_RANGE: RangeInclusive<u32> = 0xD800..=0xDFFF;
            let example = src.any_of(
                "example",
                make::int_in_range(char::MIN as u32..=char::MAX as u32)
                    .filter_assume(|u| !SURROGATE_RANGE.contains(u))
                    .map(|u| char::from_u32(u).unwrap()),
            );
            let c = src.as_raw().any("c", Some(&example));
            assert_eq!(c, example);
        });
    }

    #[test]
    fn char_examples() {
        print_debug_examples(&make::arbitrary::<char>(), None, Ord::cmp);
    }

    #[test]
    fn char_like_examples() {
        print_debug_examples(&make::arbitrary::<char>(), Some(&'Z'), Ord::cmp);
    }

    #[test]
    fn char_seeded_examples() {
        print_debug_examples(
            &make::arbitrary::<char>().seeded(&['Z'], true),
            None,
            Ord::cmp,
        );
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use crate::{Arbitrary as _, tests::bench_gen_next};

    #[bench]
    fn gen_char(b: &mut test::Bencher) {
        let g = char::arbitrary();
        bench_gen_next(b, &g);
    }
}
