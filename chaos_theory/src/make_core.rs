// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    cmp::Ordering,
    fmt::Debug,
    marker::{PhantomData, PhantomPinned},
    mem::MaybeUninit,
    num::NonZero,
    ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
    result::Result,
};

use crate::{Arbitrary, Generator, MaybeOwned, SourceEx, Tweak, make};

impl<T: ?Sized> Arbitrary for PhantomData<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        make::just(Self)
    }
}

impl Arbitrary for PhantomPinned {
    fn arbitrary() -> impl Generator<Item = Self> {
        make::just(Self)
    }
}

// Always generate uninitialized storage. `MaybeUninit` does not expose its initialization state,
// so a generator that sometimes initialized it would give callers no safe way to know when they
// may access or drop `T`; initialized values requiring drop would usually leak.
impl<T> Arbitrary for MaybeUninit<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        make::from_fn(|_| Self::uninit())
    }
}

impl Arbitrary for Ordering {
    fn arbitrary() -> impl Generator<Item = Self> {
        fn make_ordering(src: &mut SourceEx, example: Option<&Ordering>) -> Ordering {
            let variants = [Ordering::Less, Ordering::Equal, Ordering::Greater];
            let example_index = example.map(|e| {
                variants
                    .iter()
                    .position(|v| v == e)
                    .expect("internal error: invalid Ordering")
            });
            let variant_ix = src
                .as_mut()
                .choose_index(variants.len(), example_index, Tweak::None);
            variants[variant_ix]
        }

        make::from_next(make_ordering)
    }
}

#[derive(Debug)]
struct Range_<GS, GE> {
    start: GS,
    end: GE,
}

impl<GS: Generator, GE: Generator<Item = GS::Item>> Generator for Range_<GS, GE> {
    type Item = Range<GS::Item>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let start = src.any_of("start", &self.start, example.map(|r| &r.start));
        let end = src.any_of("end", &self.end, example.map(|r| &r.end));
        Range { start, end }
    }
}

impl<T: Arbitrary> Arbitrary for Range<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        Range_ {
            start: T::arbitrary(),
            end: T::arbitrary(),
        }
    }
}

impl<T: Arbitrary> Arbitrary for RangeFrom<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        T::arbitrary().map_reversible(
            |start| Self { start },
            |r| Some(MaybeOwned::Borrowed(&r.start)),
        )
    }
}

impl Arbitrary for RangeFull {
    fn arbitrary() -> impl Generator<Item = Self> {
        make::just(Self)
    }
}

#[derive(Debug)]
struct RangeInclusive_<GS, GE> {
    start: GS,
    end: GE,
}

impl<GS: Generator, GE: Generator<Item = GS::Item>> Generator for RangeInclusive_<GS, GE> {
    type Item = RangeInclusive<GS::Item>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let start = src.any_of("start", &self.start, example.map(RangeInclusive::start));
        let end = src.any_of("end", &self.end, example.map(RangeInclusive::end));
        RangeInclusive::new(start, end)
    }
}

impl<T: Arbitrary> Arbitrary for RangeInclusive<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        RangeInclusive_ {
            start: T::arbitrary(),
            end: T::arbitrary(),
        }
    }
}

impl<T: Arbitrary> Arbitrary for RangeTo<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        T::arbitrary().map_reversible(|end| Self { end }, |r| Some(MaybeOwned::Borrowed(&r.end)))
    }
}

impl<T: Arbitrary> Arbitrary for RangeToInclusive<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        T::arbitrary().map_reversible(|end| Self { end }, |r| Some(MaybeOwned::Borrowed(&r.end)))
    }
}

#[derive(Debug)]
struct Bound_<G> {
    elem: G,
}

impl<G: Generator> Generator for Bound_<G> {
    type Item = Bound<G::Item>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_index = example.map(|e| match e {
            Bound::Included(_) => 0,
            Bound::Excluded(_) => 1,
            Bound::Unbounded => 2,
        });

        let variants = &["Included", "Excluded", "Unbounded"];
        let variants_num = NonZero::new(variants.len()).expect("internal error: no variants");
        src.select(
            "<bound>",
            example_index,
            variants_num,
            |ix| variants[ix],
            |src, variant, _ix| match variant {
                "Included" => {
                    let example_elem = match example {
                        Some(Bound::Included(elem)) => Some(elem),
                        _ => None,
                    };
                    Bound::Included(self.elem.next(src, example_elem))
                }
                "Excluded" => {
                    let example_elem = match example {
                        Some(Bound::Excluded(elem)) => Some(elem),
                        _ => None,
                    };
                    Bound::Excluded(self.elem.next(src, example_elem))
                }
                "Unbounded" => Bound::Unbounded,
                _ => unreachable!(),
            },
        )
    }
}

impl<T: Arbitrary> Arbitrary for Bound<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        Bound_ {
            elem: T::arbitrary(),
        }
    }
}

#[derive(Debug)]
struct Option_<G> {
    elem: G,
}

impl<G: Generator> Generator for Option_<G> {
    type Item = Option<G::Item>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        src.maybe("<option>", example.map(Option::is_some), |src| {
            self.elem.next(src, example.and_then(|e| e.as_ref()))
        })
    }
}

impl<T: Arbitrary> Arbitrary for Option<T> {
    fn arbitrary() -> impl Generator<Item = Self> {
        option(T::arbitrary())
    }
}

/// Create an [`Option`] generator.
pub fn option<T: Debug>(elem: impl Generator<Item = T>) -> impl Generator<Item = Option<T>> {
    Option_ { elem }
}

/// Create an [`Option`] generator that always generates `Some` values.
pub fn some<T: Debug>(elem: impl Generator<Item = T>) -> impl Generator<Item = Option<T>> {
    elem.map_reversible(Option::Some, |t| t.as_ref().map(MaybeOwned::Borrowed))
}

/// Create an [`Option`] generator that always generates `None` values.
pub fn none<T: Debug>() -> impl Generator<Item = Option<T>> {
    ().map_reversible(|()| Option::None, |_| Option::Some(MaybeOwned::Owned(())))
}

#[derive(Debug)]
struct Result_<GT, GE> {
    ok: GT,
    err: GE,
}

impl<GT: Generator, GE: Generator> Generator for Result_<GT, GE> {
    type Item = Result<GT::Item, GE::Item>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_index = example.map(|e| match e {
            Ok(_) => 0,
            Err(_) => 1,
        });

        let variants = &["Ok", "Err"];
        let variants_num = NonZero::new(variants.len()).expect("internal error: no variants");
        src.select(
            "<result>",
            example_index,
            variants_num,
            |ix| variants[ix],
            |src, variant, _ix| match variant {
                "Ok" => {
                    let example_ok = match example {
                        Some(Ok(ok)) => Some(ok),
                        _ => None,
                    };
                    Ok(self.ok.next(src, example_ok))
                }
                "Err" => {
                    let example_err = match example {
                        Some(Err(err)) => Some(err),
                        _ => None,
                    };
                    Err(self.err.next(src, example_err))
                }
                _ => unreachable!(),
            },
        )
    }
}

impl<T: Arbitrary, E: Arbitrary> Arbitrary for Result<T, E> {
    fn arbitrary() -> impl Generator<Item = Self> {
        result(T::arbitrary(), E::arbitrary())
    }
}

/// Create a [`Result`] generator.
pub fn result<T: Debug, E: Debug>(
    ok: impl Generator<Item = T>,
    err: impl Generator<Item = E>,
) -> impl Generator<Item = Result<T, E>> {
    Result_ { ok, err }
}

/// Create a [`Result`] generator that always generates `Ok` values.
pub fn ok<T: Debug, E: Debug>(g: impl Generator<Item = T>) -> impl Generator<Item = Result<T, E>> {
    g.map_reversible(Result::Ok, |r| r.as_ref().ok().map(MaybeOwned::Borrowed))
}

/// Create a [`Result`] generator that always generates `Err` values.
pub fn err<T: Debug, E: Debug>(e: impl Generator<Item = E>) -> impl Generator<Item = Result<T, E>> {
    e.map_reversible(Result::Err, |r| r.as_ref().err().map(MaybeOwned::Borrowed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        check,
        tests::{print_debug_examples, prop_smoke},
    };

    #[test]
    fn ordering_smoke() {
        check(|src| {
            prop_smoke(src, "ordering", make::arbitrary::<Ordering>());
        });
    }

    #[test]
    fn range_smoke() {
        check(|src| {
            prop_smoke(src, "range", make::arbitrary::<Range<i8>>());
            prop_smoke(src, "range_from", make::arbitrary::<RangeFrom<i8>>());
            prop_smoke(src, "range_full", make::arbitrary::<RangeFull>());
            prop_smoke(
                src,
                "range_inclusive",
                make::arbitrary::<RangeInclusive<i8>>(),
            );
            prop_smoke(src, "range_to", make::arbitrary::<RangeTo<i8>>());
            prop_smoke(
                src,
                "range_to_inclusive",
                make::arbitrary::<RangeToInclusive<i8>>(),
            );
        });
    }

    #[test]
    fn bound_smoke() {
        check(|src| {
            prop_smoke(src, "bound", make::arbitrary::<Bound<i8>>());
        });
    }

    #[test]
    fn option_i8_examples() {
        let g = option(i8::arbitrary());
        print_debug_examples(g, None, Ord::cmp);
    }

    #[test]
    fn option_smoke() {
        check(|src| {
            prop_smoke(src, "option", option(i8::arbitrary()));
            prop_smoke(src, "some", some(i8::arbitrary()));
            prop_smoke(src, "none", none::<i8>());
        });
    }

    #[test]
    fn result_smoke() {
        check(|src| {
            prop_smoke(src, "result", result(i8::arbitrary(), i32::arbitrary()));
            prop_smoke(src, "ok", ok::<_, i32>(i8::arbitrary()));
            prop_smoke(src, "err", err::<i8, _>(i32::arbitrary()));
        });
    }
}

#[cfg(all(test, feature = "_bench"))]
mod benches {
    use crate::{Arbitrary as _, tests::bench_gen_next};

    #[bench]
    fn option_u64(b: &mut test::Bencher) {
        let g = Option::<u64>::arbitrary();
        bench_gen_next(b, &g);
    }

    #[bench]
    fn result_i64_u64(b: &mut test::Bencher) {
        let g = Result::<i64, u64>::arbitrary();
        bench_gen_next(b, &g);
    }
}
