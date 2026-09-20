// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::vec::Vec;
use core::{fmt::Debug, ops::Range, ops::RangeBounds};

use crate::{
    Effect, Generator, OptionExt as _, SourceEx, generator::UNABLE_GENERATE_UNIQUE, make,
    range::SizeRange,
};

pub(crate) fn shuffle_impl<T>(src: &mut SourceEx, example: Option<&[usize]>, values: &mut [T]) {
    let len = values.len();
    let max_swaps = len.saturating_sub(1);
    let swaps = example.map(|order| permutation_swaps(len, order));
    // Prefer a complete traversal when generating, but let replay reduce its size.
    let count = swaps
        .as_ref()
        .map(Vec::len)
        .or_else(|| src.as_ref().is_fresh().then_some(max_swaps));
    let example = count.map(|count| (0..count).map(|i| swaps.as_ref().map(|s| s[i])));
    let mut step = 0;
    src.repeat_finite(
        "<swaps>",
        example,
        SizeRange::new_raw(0, max_swaps),
        |_| (),
        |(), src, example| {
            let example = example.flatten();
            // Record the start as well as the distance, so removing earlier swaps
            // does not relocate other operations.
            let start = example
                .map(|(start, _)| start)
                .or_else(|| src.as_ref().is_fresh().then_some(step));
            let start = src.any_of("start", make::index(max_swaps), start.as_ref());
            // Generate "distance" instead of "end" for better natural shrinking.
            let distance = src.any_of(
                "distance",
                make::index(len - start),
                example.map(|(_, distance)| distance).as_ref(),
            );
            values.swap(start, start + distance);
            step += 1;
            Effect::Success
        },
    )
    .expect("internal error: shuffle steps should not fail");
}

fn permutation_swaps(len: usize, example: &[usize]) -> Vec<(usize, usize)> {
    let mut order: Vec<_> = (0..len).collect();
    let mut positions = order.clone();
    let mut swaps = Vec::with_capacity(len);
    for (i, &wanted) in example.iter().take(len).enumerate() {
        let Some(&j) = positions.get(wanted) else {
            continue;
        };
        // Keep earlier positions fixed, including when an index is repeated.
        if j > i {
            swaps.push((i, j - i));
            order.swap(i, j);
            positions[order[i]] = i;
            positions[order[j]] = j;
        }
    }
    swaps
}

#[derive(Debug)]
struct Permutation {
    len: usize,
}

impl Generator for Permutation {
    type Item = Vec<usize>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let mut order: Vec<_> = (0..self.len).collect();
        shuffle_impl(src, example.map(Vec::as_slice), &mut order);
        order
    }
}

/// Create a generator of permutations of `0..len`.
///
/// Zero `len` produces an empty vector.
///
/// Use [`Source::shuffle`](crate::Source::shuffle) to shuffle a slice in place.
///
/// ```
/// use chaos_theory::{check, make};
///
/// check(|src| {
///     let values = ["first", "second", "third"];
///     let order = src.any_of("order", make::permutation(values.len()));
///     let reordered: Vec<_> = order.into_iter().map(|i| values[i]).collect();
///     assert_eq!(reordered.len(), values.len());
/// });
/// ```
pub fn permutation(len: usize) -> impl Generator<Item = Vec<usize>> {
    Permutation { len }
}

#[derive(Debug)]
struct Subsequence {
    len: usize,
    size: SizeRange,
}

impl Generator for Subsequence {
    type Item = Vec<usize>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        next_indices(src, self.len, self.size, example.map(|e| e.iter().copied()))
    }
}

/// Create a generator of strictly increasing indices in `0..len`.
///
/// The result may be empty.
///
/// ```
/// use chaos_theory::{check, make};
///
/// check(|src| {
///     let values = ["first", "second", "third"];
///     let keep = src.any_of("keep", make::subsequence(values.len()));
///     let selected: Vec<_> = keep.into_iter().map(|i| values[i]).collect();
///     assert!(selected.len() <= values.len());
/// });
/// ```
pub fn subsequence(len: usize) -> impl Generator<Item = Vec<usize>> {
    subsequence_n(len, ..)
}

/// Create a subsequence index generator with the specified number of elements.
///
/// Intersects the requested size range with `0..=len`. For example,
/// `subsequence_n(3, 1..=10)` produces between one and three indices.
///
/// # Panics
///
/// Panics if the size range is invalid or contains no feasible size, such as
/// `subsequence_n(0, 1..)`.
pub fn subsequence_n(
    len: usize,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = Vec<usize>> {
    Subsequence {
        len,
        size: bound_size(size, 0, len),
    }
}

#[derive(Debug)]
struct Chunks {
    len: usize,
    size: SizeRange,
}

impl Generator for Chunks {
    type Item = Vec<Range<usize>>;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example = example.map(|e| {
            e[..e.len().saturating_sub(1)]
                .iter()
                // Keep zero endpoints out of range for the index generator.
                .map(|r| r.end.wrapping_sub(1))
        });
        let cuts = next_indices(
            src,
            self.len.saturating_sub(1),
            SizeRange::new_raw(
                self.size.min.saturating_sub(1),
                self.size.max.saturating_sub(1),
            ),
            example,
        );
        let mut start = 0;
        cuts.into_iter()
            .map(|i| i + 1)
            .chain((self.len != 0).then_some(self.len))
            .map(|end| {
                let range = start..end;
                start = end;
                range
            })
            .collect()
    }
}

/// Create a generator of adjacent, nonempty ranges covering `0..len`.
///
/// Shrinking merges adjacent chunks, moving toward one range covering the
/// whole input. An empty input produces no chunks.
///
/// ```
/// use chaos_theory::{check, make};
///
/// check(|src| {
///     let bytes = b"streaming input";
///     let chunks = src.any_of("chunks", make::chunks(bytes.len()));
///     let mut received = Vec::new();
///     for range in chunks {
///         received.extend_from_slice(&bytes[range]);
///     }
///     assert_eq!(received, bytes);
/// });
/// ```
pub fn chunks(len: usize) -> impl Generator<Item = Vec<Range<usize>>> {
    chunks_n(len, ..)
}

/// Create a chunk generator with the specified number of chunks.
///
/// `size` constrains the chunk count, not individual chunk lengths. Feasible
/// counts are `1..=len` for nonempty input, and zero for empty input.
/// Shrinking respects the requested minimum chunk count.
///
/// # Panics
///
/// Panics if the size range is invalid or contains no feasible chunk count.
pub fn chunks_n(
    len: usize,
    size: impl RangeBounds<usize>,
) -> impl Generator<Item = Vec<Range<usize>>> {
    Chunks {
        len,
        size: bound_size(size, usize::from(len != 0), len),
    }
}

#[track_caller]
fn bound_size(size: impl RangeBounds<usize>, min: usize, max: usize) -> SizeRange {
    let size = SizeRange::new(size);
    let (min, max) = (size.min.max(min), size.max.min(max));
    assert!(min <= max, "size range contains no feasible size");
    SizeRange::new_raw(min, max)
}

fn next_indices(
    src: &mut SourceEx,
    len: usize,
    size: SizeRange,
    example: Option<impl ExactSizeIterator<Item = usize>>,
) -> Vec<usize> {
    let mut available: Vec<_> = (0..len).collect();
    let mut positions = available.clone();
    let result = src.repeat_finite(
        "<indices>",
        example,
        size,
        Vec::with_capacity,
        |indices, src, example| {
            // Fresh examples come from the available pool. Replay reuses absolute
            // indices, so deleting one choice preserves the remaining selections.
            let example = example.or_else(|| src.as_mut().choose_example(1.0, &available).copied());
            let i = src.any_of("index", make::index(len), example.as_ref());
            let position = positions[i];
            if position == usize::MAX {
                return Effect::Noop;
            }
            available.swap_remove(position);
            if let Some(&moved) = available.get(position) {
                positions[moved] = position;
            }
            positions[i] = usize::MAX;
            indices.push(i);
            Effect::Success
        },
    );
    let mut indices = result.assume_some_msg(UNABLE_GENERATE_UNIQUE);
    indices.sort_unstable();
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, tests::prop_smoke};

    #[test]
    fn sequence_smoke() {
        check(|src| {
            let len = src.any_of("len", make::size(..));
            prop_smoke(src, "permutation", permutation(len));
            prop_smoke(src, "subsequence", subsequence(len));
            prop_smoke(src, "chunks", chunks(len));
        });
    }

    #[test]
    fn sequence_invariants() {
        check(|src| {
            let len = src.any_of("len", make::int_in(..100)); // Make sure to cover north of MAX_SIZE
            let count = src.any_of("count", make::index(len + 1));
            let mut order = src.any_of("order", permutation(len));
            order.sort_unstable();
            assert_eq!(order, (0..len).collect::<Vec<_>>());
            let selected = src.any_of("selected", subsequence_n(len, count..=count));
            assert_eq!(selected.len(), count);
            assert!(selected.iter().all(|&i| i < len));
            assert!(selected.windows(2).all(|w| w[0] < w[1]));
            let count = count.max(usize::from(len != 0));
            let ranges = src.any_of("ranges", chunks_n(len, count..=count));
            assert_eq!(ranges.len(), count);
            let mut end = 0;
            for range in ranges {
                assert_eq!(range.start, end);
                assert!(range.start < range.end);
                end = range.end;
            }
            assert_eq!(end, len);
        });
    }
}
