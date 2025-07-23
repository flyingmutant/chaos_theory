// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::time::Duration;
use std::{sync::Once, time::Instant};

use crate::{Set, tape::Tape, unwind::PanicInfo};

pub(crate) fn reduce_tape(
    tape: Tape,
    info: PanicInfo,
    limit: Duration,
    trial: impl FnMut(Tape) -> (Tape, Option<PanicInfo>),
) -> (Tape, PanicInfo) {
    debug_assert!(!info.invalid_data);
    let mut r = Reducer {
        tape,
        info,
        start: Instant::now(),
        limit,
        trial,
        trials: 0,
        reductions: 0,
        timed_out: false,
        cache: Set::default(),
        flaky_log_once: Once::new(),
    };
    r.reduce();
    if r.timed_out && !limit.is_zero() {
        let (trials, reductions, elapsed) = (r.trials, r.reductions, r.start.elapsed());
        eprintln!(
            "[chaos_theory] stopping test case reduction early after {trials} attempts ({reductions} reductions), elapsed {elapsed:?} with time limit of {limit:?}"
        );
    }
    // Run trial one more time, so that any side-effects it has, are from the tape we are returning
    // and not some intermediary one.
    let (tape_out, info_out) = (r.trial)(r.tape.clone());
    // Return the output tape since it contains valid metadata.
    debug_assert!(tape_out.has_meta());
    // Try to return matching info, if possible (if test is not flaky).
    if let Some(info_out) = info_out {
        (tape_out, info_out)
    } else {
        (tape_out, r.info)
    }
}

struct Reducer<F> {
    tape: Tape,
    info: PanicInfo,
    start: Instant,
    limit: Duration,
    trial: F,
    trials: usize,
    reductions: usize,
    timed_out: bool,
    cache: Set<u64>,
    flaky_log_once: Once,
}

impl<F: FnMut(Tape) -> (Tape, Option<PanicInfo>)> Reducer<F> {
    fn reduce(&mut self) {
        // Optimistically get rid of all noop data first, before we run the first pass.
        let r = self.try_incorporate(self.tape.clone(), false);
        if r.is_err() {
            // Early timeout.
            self.timed_out = true;
            return;
        }
        // Things we don't do:
        // - try to match the repeat size with an integer before and reduce both simultaneously
        //   - more generally, try to match equal values and reduce them simultaneously
        // - try re-order the sequences
        // - try to unset or reorder bits in numbers
        // - try to re-distribute integer amounts
        // - for recursive data, replace values with sub-values.
        //
        // All we do is hierarchically reduce the sequences in the tree and reduce numbers using binary search.
        let passes = [Self::pass_reduce_tree];
        loop {
            let reductions_before = self.reductions;
            for pass in passes {
                let r = pass(self);
                if r.is_err() {
                    // Timeout.
                    self.timed_out = true;
                    return;
                }
            }
            if self.reductions == reductions_before {
                // No pass can make any progress.
                return;
            }
        }
    }

    fn try_incorporate(&mut self, t: Tape, skip_non_smaller: bool) -> Result<bool, ()> {
        // Special-case zero limit because `elapsed()` can return zero as well.
        if self.limit.is_zero() || self.start.elapsed() > self.limit {
            return Err(());
        }
        if skip_non_smaller && !t.smaller_choices(&self.tape) {
            // Most of the time, the tape we are trying to incorporate is smaller
            // than the current, but not always. While reducing the tape, during one pass
            // we operate on one tree, trying to cut it down. But when incorporating
            // variants of the tree, we may get an even shorter output tape (even without discard).
            // We'll remember it, and it can be smaller than many other variants from this tree.
            // However, on the next iteration, we'll build new tree from the smallest tape seen.
            return Ok(false);
        }
        if !self.cache.insert(t.hash()) {
            // We've already tried this tape; do nothing.
            return Ok(false);
        }
        self.trials += 1;
        let (t_out, t_info) = (self.trial)(t);
        // `t_out` might be different from `t` because `t` is a result of (blind) reduction.
        if let Some(t_info) = t_info
            && t_info.same_location(&self.info)
        {
            let t_best = if t_out.has_noop() {
                let t_disc = t_out.discard_noop();
                // Try to make sure discarded data did not affect the test result.
                self.trials += 1;
                let (t_out2, t_info2) = (self.trial)(t_disc.clone());
                if t_out2 == t_disc && t_info2.as_ref() == Some(&t_info) {
                    t_disc
                } else {
                    self.flaky_log_once.call_once(|| {
                            eprintln!("[chaos_theory] test result changed after removing data marked as noop; either test ran out of RNG budget, is flaky or some repeat() calls lie about state changes");
                        });
                    t_out
                }
            } else {
                t_out
            };
            if t_best.smaller_choices(&self.tape) {
                self.reductions += 1;
                self.tape = t_best;
                self.info = t_info;
                return Ok(true);
            }
        }
        // We simply forget about the tape if the error location is different.
        Ok(false)
    }

    fn pass_reduce_tree(&mut self) -> Result<(), ()> {
        let mut tree = self.tape.clone().into_tree();
        let (_removed, _reduced, early_exit) = reduce_tree(&mut tree, |t| {
            // Don't ignore noop scopes, as they might affect the result.
            let tape = t.to_tape(false);
            // Note: after incorporating candidate tape, self.tape may be less than next tapes we'll generate.
            // However, we'll not build a new tree from the tape unless we are finishing trying to reduce this one.
            self.try_incorporate(tape, true).ok()
        });
        if early_exit { Err(()) } else { Ok(()) }
    }
}

pub(crate) trait Tree {
    type NodeId: Copy;
    type Child: TreeNodeChild<Self::NodeId>;

    fn root(&self) -> Option<Self::NodeId>;
    fn children_num(&self, node_id: Self::NodeId) -> usize;
    fn child(&self, node_id: Self::NodeId, ix: usize) -> Self::Child;
    fn child_replace(&mut self, node_id: Self::NodeId, ix: usize, child: &Self::Child);
}

pub(crate) trait TreeNodeChild<NodeId>: Sized {
    fn reduce(self, accept: impl FnMut(&Self) -> Option<bool>) -> (Self, usize, usize, bool);
    fn extend_vec(self, v: &mut Vec<NodeId>);
}

pub(crate) trait Seq: Sized {
    fn mask(&self, begin: usize, end: usize) -> Option<(Self, usize)>;
    fn size_min(&self) -> usize;
    fn size_masked(&self) -> usize;
    fn size_total(&self) -> usize;
}

// Hierarchical version of the ddmin-like algorithm.
//
// HDDr algorithm (DF BW, depth-first with backwards visitation, variant: pop-last + append-forward)
// "HDDr: a recursive variant of the hierarchical Delta debugging algorithm"
//
// procedure HDDr′(root_node)
//     queue ← ⟨root_node⟩
//     while queue ⟨⟩ do
//         current_node ← pop(queue)
//         nodes ← tagChildren(current_node)
//         min_config ← ddmin(nodes)
//         pruneChildren(current_node, min_config)
//         append(queue, min_config)
//     end while
// end procedure
//
fn reduce_tree<T: Tree>(
    t: &mut T,
    mut accept: impl FnMut(&T) -> Option<bool>,
) -> (usize, usize, bool) {
    let mut queue = Vec::new();
    queue.extend(t.root());
    let mut removed_total = 0;
    let mut reduced_total = 0;
    let mut level_children = Vec::new();
    let mut early_exit = false;
    'main_loop: while let Some(node) = queue.pop() {
        let children_num = t.children_num(node);
        for ix_fwd in 0..children_num {
            // Iterate from the right to the left, with intuition being that we want to try to remove the end of the sequence first.
            let ix = children_num - ix_fwd - 1;
            let child = t.child(node, ix);
            let (child_min, removed, reduced, child_early_exit) = child.reduce(|c| {
                // Note: we are modifying the tree in-place.
                t.child_replace(node, ix, c);
                accept(t)
            });
            t.child_replace(node, ix, &child_min);
            removed_total += removed;
            reduced_total += reduced;
            if child_early_exit {
                early_exit = true;
                break 'main_loop;
            }
            // Push groups in the reverse order.
            level_children.push(child_min);
        }
        let children = core::mem::take(&mut level_children);
        // Extend the queue with groups in the normal order.
        for child in children.into_iter().rev() {
            child.extend_vec(&mut queue);
        }
    }
    (removed_total, reduced_total, early_exit)
}

// Simple homegrown ddmin wannabe.
//
// CDD algorithm from "Deep Dive into Probabilistic Delta Debugging: Insights and Simplifications"
// is below (for reference). See also:
// - OPDD from "Avoiding the Familiar to Speed Up Test Case Reduction"
// - Lithium from https://github.com/MozillaSecurity/lithium
//
// r ← 0
// do
//     s ← ComputeSize(r, p0)
//     subsets ← Partition(L, s)
//     foreach subset ∈ subsets do
//         temp ← L \ subset
//         if ψ(temp) is true then
//             L ← temp
//     r ← r + 1
// while s > 1
// return L
//
// Function ComputeSize(r, p0):
//     s0 ← ⌊−1/ ln(1−p0)⌋
//     s  ← ⌊s0 × 0.632^r⌋
//     return s
//
pub(crate) fn reduce_seq<S: Seq>(
    mut s: S,
    mut accept: impl FnMut(&S) -> Option<bool>,
) -> (S, usize, bool) {
    let size_total = s.size_total();
    let size_orig = size_total - s.size_masked();
    let size_min = s.size_min();
    let mut subset_size = size_total;
    let mut remaining = size_orig;
    let mut early_exit = false;
    'main_loop: while subset_size > 0 && remaining > size_min {
        let mut n = 0;
        loop {
            let begin = (subset_size * n).min(size_total);
            let end = (subset_size * (n + 1)).min(size_total);
            if begin == end {
                break;
            }
            // Iterate from the right to the left, with intuition being that we want to try to remove the end of the sequence first.
            let rev_begin = size_total - end;
            let rev_end = size_total - begin;
            debug_assert!(rev_begin < rev_end);
            if let Some((c, masked)) = s.mask(rev_begin, rev_end)
                && remaining - masked >= size_min
            {
                let ok = accept(&c);
                if let Some(ok) = ok {
                    if ok {
                        remaining -= masked;
                        s = c;
                    }
                } else {
                    early_exit = true;
                    break 'main_loop;
                }
            }
            n += 1;
        }
        subset_size /= 2;
    }
    (s, size_orig - remaining, early_exit)
}

// TODO: try to reduce by one and exit fast if we can't
pub(crate) fn reduce_num(n: u64, mut accept: impl FnMut(u64) -> Option<bool>) -> (u64, bool, bool) {
    // Binary search does badly when the property is non-monotonic, failing to explore smaller values
    // after the first negative query. Force-check the small values first.
    if let Some((s, early_exit)) = reduce_num_small(n, &mut accept) {
        return (s, s < n, early_exit);
    }
    let (m, early_exit) = reduce_num_binsearch(n, &mut accept);
    (m, m < n, early_exit)
}

fn reduce_num_small(n: u64, mut accept: impl FnMut(u64) -> Option<bool>) -> Option<(u64, bool)> {
    let small = [0, 1, 2, 3];
    for s in small {
        if s >= n {
            return Some((n, false));
        }
        let ok = accept(s);
        if let Some(ok) = ok {
            if ok {
                return Some((s, false));
            }
        } else {
            return Some((n, true));
        }
    }
    None
}

fn reduce_num_binsearch(n: u64, mut accept: impl FnMut(u64) -> Option<bool>) -> (u64, bool) {
    // We assume accept(n) = true.
    let (mut lo, mut hi) = (0, n);
    let mut early_exit = false;
    while lo < hi {
        let h = lo + (hi - lo) / 2;
        debug_assert!(h < hi);
        let ok = accept(h);
        if let Some(ok) = ok {
            if ok {
                hi = h;
            } else {
                lo = h + 1;
            }
        } else {
            early_exit = true;
            break;
        }
    }
    (hi, early_exit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Effect, Map, assume, check, make, vdbg};
    use core::num::NonZero;

    struct ToyTree {
        nodes: Map<ToyNodeId, ToyNode>,
        next_node_id: ToyNodeId,
    }

    type ToyNodeId = NonZero<u32>;

    struct ToyChild(Vec<Option<ToyNodeId>>);

    struct ToyNode {
        value: u8,
        children: Vec<ToyChild>,
    }

    impl ToyTree {
        fn new() -> Self {
            Self {
                nodes: Map::default(),
                next_node_id: ToyNodeId::new(1).unwrap(),
            }
        }

        fn add_child(&mut self, parent_id: Option<ToyNodeId>, value: u8, new_group: bool) {
            let child_id = self.next_node_id;
            let prev = self.nodes.insert(
                child_id,
                ToyNode {
                    value,
                    children: vec![ToyChild(Vec::new())],
                },
            );
            debug_assert!(prev.is_none());
            self.next_node_id = child_id.checked_add(1).unwrap();
            if let Some(parent_id) = parent_id {
                let parent = self.nodes.get_mut(&parent_id).unwrap();
                if new_group {
                    parent.children.push(ToyChild(Vec::new()));
                }
                parent.children.last_mut().unwrap().0.push(Some(child_id));
            }
        }
    }

    impl Seq for ToyChild {
        fn mask(&self, begin: usize, end: usize) -> Option<(Self, usize)> {
            let n = self.0[begin..end].iter().filter(|id| id.is_some()).count();
            if n == 0 {
                None
            } else {
                let mut v = self.0.clone();
                v[begin..end].fill(Option::None);
                Some((Self(v), n))
            }
        }

        fn size_min(&self) -> usize {
            self.size_total() / 2 // something more interesting than just "0"
        }

        fn size_masked(&self) -> usize {
            self.0.iter().filter(|c| c.is_none()).count()
        }

        fn size_total(&self) -> usize {
            self.0.len()
        }
    }

    impl TreeNodeChild<ToyNodeId> for ToyChild {
        fn reduce(self, accept: impl FnMut(&Self) -> Option<bool>) -> (Self, usize, usize, bool) {
            let (r, removed, early_exit) = reduce_seq(self, accept);
            (r, removed, 0, early_exit)
        }

        fn extend_vec(self, v: &mut Vec<ToyNodeId>) {
            v.extend(self.0.into_iter().flatten());
        }
    }

    impl Tree for ToyTree {
        type NodeId = ToyNodeId;
        type Child = ToyChild;

        fn root(&self) -> Option<Self::NodeId> {
            if self.nodes.is_empty() {
                None
            } else {
                Some(ToyNodeId::new(1).unwrap())
            }
        }

        fn children_num(&self, node_id: Self::NodeId) -> usize {
            self.nodes[&node_id].children.len()
        }

        fn child(&self, node_id: Self::NodeId, ix: usize) -> Self::Child {
            ToyChild(self.nodes[&node_id].children[ix].0.clone())
        }

        fn child_replace(&mut self, node_id: Self::NodeId, ix: usize, child: &Self::Child) {
            let node = self.nodes.get_mut(&node_id).unwrap();
            let node_child = &mut node.children[ix];
            debug_assert_eq!(node_child.0.len(), child.0.len());
            node_child.0 = child.0.clone();
        }
    }

    #[test]
    fn reduce_toy_tree() {
        fn accept(t: &ToyTree) -> bool {
            sum(t) != 0
        }

        fn sum(t: &ToyTree) -> u64 {
            fn sum_subtree(t: &ToyTree, n: ToyNodeId) -> u64 {
                let node = &t.nodes[&n];
                let mut s = u64::from(node.value);
                for group in &node.children {
                    for child in group.0.iter().flatten() {
                        s += sum_subtree(t, *child);
                    }
                }
                s
            }

            if let Some(root_id) = t.root() {
                sum_subtree(t, root_id)
            } else {
                0
            }
        }

        check(|src| {
            let mut t = ToyTree::new();
            src.repeat("add child", |src| {
                let parent_id = src.any_of("parent_id", make::int_in_range(..t.next_node_id.get()));
                let value = src.any("value");
                let new_group = src.any("new_group");
                if parent_id == 0 {
                    if !t.nodes.is_empty() {
                        return Effect::Noop;
                    }
                    t.add_child(None, value, new_group);
                } else {
                    let parent_id = ToyNodeId::new(parent_id).unwrap();
                    t.add_child(Some(parent_id), value, new_group);
                }
                Effect::Success
            });
            assume!(accept(&t));
            let sum_before = sum(&t);
            let (mut removed, mut reduced, mut early_exit) = (1, 1, false);
            while removed > 0 && reduced > 0 && !early_exit {
                (removed, reduced, early_exit) = reduce_tree(&mut t, |t| Some(accept(t)));
            }
            let sum_after = sum(&t);
            assert!(accept(&t));
            assert!(sum_after <= sum_before);
            vdbg!(src, (sum_before, sum_after));
        });
    }

    struct ToySeq(Vec<u8>, usize);

    impl Seq for ToySeq {
        fn mask(&self, begin: usize, end: usize) -> Option<(Self, usize)> {
            let n = self.0[begin..end].iter().filter(|n| **n != 0).count();
            if n == 0 {
                None
            } else {
                let mut v = self.0.clone();
                v[begin..end].fill(0);
                Some((Self(v, self.1), n))
            }
        }

        fn size_min(&self) -> usize {
            self.1
        }

        #[expect(clippy::naive_bytecount)]
        fn size_masked(&self) -> usize {
            self.0.iter().filter(|n| **n == 0).count()
        }

        fn size_total(&self) -> usize {
            self.0.len()
        }
    }

    #[test]
    fn reduce_toy_seq() {
        fn accept(s: &ToySeq) -> bool {
            sum(s) % 2 != 0
        }

        fn sum(s: &ToySeq) -> u64 {
            s.0.iter().map(|n| u64::from(*n)).sum::<u64>()
        }

        check(|src| {
            let s: Vec<u8> = src.any("s");
            let size = src.any_of("size", make::int_in_range(..=s.len()));
            let mut s = ToySeq(s, size);
            let can_remove = s.size_total() - s.size_masked() >= s.size_min();
            assume!(accept(&s));
            let sum_before = sum(&s);
            let (mut removed, mut early_exit) = (1, false);
            while removed > 0 && !early_exit {
                let (masked, total) = (s.size_masked(), s.size_total());
                (s, removed, early_exit) = reduce_seq(s, |s| Some(accept(s)));
                assert!(removed <= total - masked);
            }
            let sum_after = sum(&s);
            vdbg!(src, (&s.0, sum_before, sum_after));
            assert!(accept(&s));
            assert!(sum_after <= sum_before);
            if can_remove {
                assert!(s.size_total() - s.size_masked() >= s.size_min());
            }
        });
    }

    #[test]
    fn reduce_toy_num() {
        check(|src| {
            let mut n: u64 = src.any("n");
            let lo: u64 = src.any("lo");
            // Condition that sometimes fail just because.
            let accept = |n| n > lo && n % 2 == 1;
            assume!(accept(n));
            let before = n;
            let mut changed = true;
            while changed {
                (n, changed, _) = reduce_num(n, |u| Some(accept(u)));
            }
            vdbg!(src, (before, n));
            assert!(accept(n));
            assert!(n <= before);
        });
    }
}
