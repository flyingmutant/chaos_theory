// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::mem::swap;

use crate::{
    Effect, MAX_SIZE,
    distrib::{Biased, temperature_scale_down},
    hash_identity::NoHashMap,
    rand::DefaultRand,
    tape::Tape,
    tape_event::{Event, ScopeKind},
    tape_mutate::{Reservoir, depth_index, mutation_points, rand_or},
};

// Note: metadata will *not* be valid after mutation.
pub(crate) fn crossover_events(
    events: &mut Vec<Event>,
    other: &[Event],
    result: &mut Vec<Event>,
    rng: &mut DefaultRand,
    t: u8,
    multi: bool,
    cache: &mut CrossoverCache,
) {
    debug_assert!(result.is_empty());
    let n = if multi { mutation_points(rng, t) } else { 1 };
    let mut want_merge_partial = true;
    let mut want_merge_full = true;
    let mut want_overwrite = true;
    let mut n_ok = 0;
    let mut last_ok = false;
    while n_ok < n {
        let actions = usize::from(want_merge_partial)
            + usize::from(want_merge_full)
            + usize::from(want_overwrite);
        if actions == 0 {
            // Should not happen.
            return;
        }
        if last_ok {
            debug_assert!(!result.is_empty());
            swap(result, events);
            result.clear();
        }
        let mut action = rng.next_below(actions);
        if want_merge_partial {
            if action == 0 {
                last_ok = crossover_merge_repeat(events, other, result, rng, t, false, cache);
                if last_ok {
                    n_ok += 1;
                } else {
                    want_merge_partial = false;
                }
                continue;
            }
            action -= 1;
        }
        if want_merge_full {
            if action == 0 {
                last_ok = crossover_merge_repeat(events, other, result, rng, t, true, cache);
                if last_ok {
                    n_ok += 1;
                } else {
                    want_merge_full = false;
                }
                continue;
            }
            action -= 1;
        }
        if want_overwrite {
            debug_assert_eq!(action, 0);
            last_ok = crossover_overwrite_section(events, other, result, rng, t, cache);
            if last_ok {
                n_ok += 1;
            } else {
                want_overwrite = false;
            }
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct CrossoverCache {
    // Note: because we only match by ID, in case of ID collision we can produce an invalid tape.
    scopes: [NoHashMap<u64, ScopeInfo>; 2],
    other_scopes: [NoHashMap<u64, ScopeInfo>; 2],
}

impl CrossoverCache {
    fn choose_first_scope_pair_as_random(&self, rng: &mut DefaultRand) -> Option<(usize, usize)> {
        let lo =
            Self::choose_first_scope_pair_as_random_impl(&self.scopes[0], &self.other_scopes[0]);
        let hi =
            Self::choose_first_scope_pair_as_random_impl(&self.scopes[1], &self.other_scopes[1]);
        rand_or(rng, lo, hi)
    }

    fn choose_first_scope_pair_as_random_impl(
        scopes: &NoHashMap<u64, ScopeInfo>,
        other_scopes: &NoHashMap<u64, ScopeInfo>,
    ) -> Option<(usize, usize)> {
        // Iteration provides random order.
        for main in scopes.values() {
            if let Some(other) = other_scopes.get(&main.id) {
                debug_assert_eq!(main.id, other.id);
                return Some((
                    main.res.choice.expect("internal error: empty reservoir"),
                    other.res.choice.expect("internal error: empty reservoir"),
                ));
            }
        }
        None
    }
}

#[derive(Debug)]
struct ScopeInfo {
    id: u64,
    res: Reservoir,
}

impl ScopeInfo {
    fn new(id: u64) -> Self {
        Self {
            id,
            res: Reservoir::default(),
        }
    }
}

// Select a random matching repeat that can be extended and merge it.
fn crossover_merge_repeat(
    events: &[Event],
    other: &[Event],
    result: &mut Vec<Event>,
    rng: &mut DefaultRand,
    t: u8,
    full_merge: bool,
    cache: &mut CrossoverCache,
) -> bool {
    debug_assert!(result.is_empty());
    fill_merge_scope_info(events, rng, true, &mut cache.scopes);
    fill_merge_scope_info(other, rng, false, &mut cache.other_scopes);
    let pair = cache.choose_first_scope_pair_as_random(rng);
    let Some((ix_main, ix_other)) = pair else {
        return false;
    };
    result.reserve(events.len() + other.len());
    result.extend_from_slice(&events[..=ix_main]);
    let Event::Size {
        size: size_main,
        min: min_main,
        max: max_main,
    } = events[ix_main + 1]
    else {
        unreachable!("internal error: must be a main size event");
    };
    let Event::Size {
        size: size_other, ..
    } = other[ix_other + 1]
    else {
        unreachable!("internal error: must be an other size event");
    };
    let max_to_insert = (max_main - size_main).min(size_other);
    debug_assert!(max_to_insert > 0);
    let max_size_combined = (size_main + max_to_insert) as usize;
    debug_assert!((min_main..=max_main).contains(&(max_size_combined as u64)));
    result.extend_from_slice(&[
        Event::Size {
            size: max_size_combined as u64,
            min: min_main,
            max: max_main,
        },
        Event::ScopeEnd,
    ]);
    // Above are optimistic size estimates, based on repeat headers. However, one or both
    // tapes may be incomplete, thus the real number of elements might be lower.
    let mut inserted = 0_usize;
    let mut remaining_main = size_main as usize;
    let mut remaining_other = size_other as usize;
    let mut i = ix_main + 3;
    let mut j = ix_other + 3;
    while inserted < max_size_combined && remaining_main + remaining_other > 0 {
        let (src, ix, remaining) = if rng.coinflip_fair() {
            (events, &mut i, &mut remaining_main)
        } else {
            (other, &mut j, &mut remaining_other)
        };
        let n = repeat_merge_amount(rng, t, (*remaining).min(max_size_combined - inserted));
        *remaining -= n;
        let keep = full_merge || inserted < min_main as usize || rng.coinflip_fair();
        if keep {
            inserted += n;
        }
        for _ in 0..n {
            // Skip all noop elements; hopefully they all were truly no-op.
            while matches!(
                src.get(*ix),
                Some(Event::ScopeStart {
                    effect: Effect::Noop,
                    ..
                })
            ) {
                *ix = Tape::find_after_scope_end(src, *ix + 1);
            }
            // If the tape is incomplete, we should stop if we've ran out of elements.
            // Note that `inserted` is still incremented for the full amount above.
            if *ix >= src.len() {
                break;
            }
            debug_assert!(matches!(
                src[*ix],
                Event::ScopeStart {
                    kind: ScopeKind::RepeatElement,
                    ..
                }
            ));
            let ix_after_end = Tape::find_after_scope_end(src, *ix + 1);
            if keep {
                result.extend_from_slice(&src[*ix..ix_after_end]);
            }
            *ix = ix_after_end;
        }
    }
    if full_merge {
        debug_assert_eq!(inserted, max_size_combined);
    } else {
        debug_assert!((min_main..=max_main).contains(&(inserted as u64)));
        let Event::Size { size, .. } = &mut result[ix_main + 1] else {
            unreachable!("internal error: can't find size event in crossover merge output");
        };
        *size = inserted as u64;
    }
    // Skip all leftover repeat elements from the events that we could not fit in.
    while matches!(
        events.get(i),
        Some(Event::ScopeStart {
            kind: ScopeKind::RepeatElement,
            ..
        })
    ) {
        i = Tape::find_after_scope_end(events, i + 1);
    }
    result.extend_from_slice(&events[i..]);
    true
}

fn repeat_merge_amount(rng: &mut DefaultRand, t: u8, mut max_size: usize) -> usize {
    if max_size == 0 {
        return 0;
    }
    max_size = max_size.min(MAX_SIZE);
    // Lower the temperature to split repeats into more parts.
    let dist = Biased::new_temperature(temperature_scale_down(t), None);
    let n = dist.sample(rng, max_size);
    n + 1
}

fn fill_merge_scope_info(
    events: &[Event],
    rng: &mut DefaultRand,
    main: bool,
    scopes: &mut [NoHashMap<u64, ScopeInfo>; 2],
) {
    scopes[0].clear();
    scopes[1].clear();
    let mut depth = 0;
    let id_seed = rng.next();
    let mut ix = 0;
    while ix < events.len() {
        let event = &events[ix];
        match event {
            Event::ScopeStart {
                id,
                kind,
                effect,
                meta: _,
            } => {
                // Don't try to merge noop events.
                if *effect == Effect::Noop {
                    ix = Tape::find_after_scope_end(events, ix + 1);
                    continue;
                }
                if *kind == ScopeKind::RepeatSize {
                    let Event::Size { size, max, .. } = events[ix + 1] else {
                        unreachable!("internal error: non-size event following repeat size");
                    };
                    let ok = if main { size < max } else { size > 0 };
                    if ok {
                        let d_ix = depth_index(depth);
                        let key = if main { *id ^ id_seed } else { *id };
                        let ent = scopes[d_ix]
                            .entry(key)
                            .or_insert_with(|| ScopeInfo::new(*id));
                        ent.res.accept(rng, ix);
                    }
                }
                depth += 1;
            }
            Event::ScopeEnd => {
                depth -= 1;
            }
            _ => {}
        }
        ix += 1;
    }
    debug_assert_eq!(depth, 0);
}

// Select a random matching scope and overwrite it and several after with consecutive scopes.
fn crossover_overwrite_section(
    events: &[Event],
    other: &[Event],
    result: &mut Vec<Event>,
    rng: &mut DefaultRand,
    t: u8,
    cache: &mut CrossoverCache,
) -> bool {
    debug_assert!(result.is_empty());
    fill_overwrite_scope_info(events, rng, true, &mut cache.scopes);
    fill_overwrite_scope_info(other, rng, false, &mut cache.other_scopes);
    let pair = cache.choose_first_scope_pair_as_random(rng);
    let Some((ix_main, ix_other)) = pair else {
        return false;
    };
    result.reserve(events.len() + other.len());
    result.extend_from_slice(&events[..ix_main]);
    let amount = overwrite_scope_amount(rng, t);
    // Note: code below duplicates tape advance logic from Tape::pop_* methods, but reusing it is kind of awkward.
    let (mut i, mut j, mut j_void, mut written) = (ix_main, ix_other, 0, 0);
    while i < events.len() && written < amount {
        let event = events[i].clone();
        match event {
            Event::ScopeStart {
                id,
                kind,
                effect,
                meta: _,
            } => {
                // Don't try to overwrite noop scopes.
                if effect == Effect::Noop {
                    i = Tape::find_after_scope_end(events, i + 1);
                    continue;
                }
                // Advance j past the scope start.
                if j_void > 0 {
                    j_void += 1;
                } else {
                    // Skip all noop scopes.
                    while matches!(
                        other.get(j),
                        Some(Event::ScopeStart {
                            effect: Effect::Noop,
                            ..
                        })
                    ) {
                        j = Tape::find_after_scope_end(other, j + 1);
                    }
                    // Try to find scope start of the same kind.
                    let matched;
                    (j, matched) = Tape::find_after_scope_start(other, kind, j);
                    if !matched {
                        j_void = 1;
                    }
                }
                // Now, i is at the scope start, j is right after scope start or in the void.
                let use_other =
                    // Substitute if we have a hard match.
                    // Don't try to substitute RepeatSize and SelectIndex, which preserves the shape of the tape.
                    // Note: since we match by ID, different select variants would not be substituted. Is this what we want?
                    if kind != ScopeKind::RepeatSize && kind != ScopeKind::SelectVariant && j_void == 0
                        && matches!(other[j-1], Event::ScopeStart { id: j_id, kind: j_kind, effect: Effect::Success | Effect::Change, .. } if id == j_id && kind == j_kind)
                    {
                        let j_after_end = Tape::find_after_scope_end(other, j);
                        Some((j - 1, j_after_end))
                    } else {
                        None
                };
                if let Some((j_start, j_after_end)) = use_other {
                    let i_after_end = Tape::find_after_scope_end(events, i + 1);
                    if i == 0 && i_after_end == events.len() {
                        // Avoid changing the root scope.
                        result.push(event);
                        i += 1;
                    } else {
                        result.extend_from_slice(&other[j_start..j_after_end]);
                        i = i_after_end;
                        j = j_after_end;
                        written += 1;
                    }
                } else {
                    result.push(event);
                    i += 1;
                }
                // Now, either we have pushed only scope start and advanced i and j by 1,
                // or we have pushed a full scope from other and advanced i and j by relative scope length.
            }
            Event::ScopeEnd => {
                result.push(event);
                i += 1;
                // Advance j past the scope end.
                if j_void > 0 {
                    j_void -= 1;
                } else if j < other.len() {
                    j = Tape::find_after_scope_end(other, j);
                }
            }
            Event::Size { .. } | Event::Index { .. } | Event::Value { .. } => {
                result.push(event);
                i += 1;
                // Advance j past the choice.
                if j_void == 0
                    && matches!(
                        other.get(j),
                        Some(Event::Size { .. } | Event::Index { .. } | Event::Value { .. })
                    )
                {
                    j += 1;
                }
            }
            Event::Meta(..) => {
                // Discard the event.
                // Note that meta events can still be copied into the output elsewhere.
                i += 1;
                if j_void == 0 && matches!(other.get(j), Some(Event::Meta(..))) {
                    j += 1;
                }
            }
        }
    }
    result.extend_from_slice(&events[i..]);
    true
}

fn overwrite_scope_amount(rng: &mut DefaultRand, t: u8) -> usize {
    let dist = Biased::new_temperature(t, None);
    let n = dist.sample(rng, MAX_SIZE);
    n + 1
}

fn fill_overwrite_scope_info(
    events: &[Event],
    rng: &mut DefaultRand,
    main: bool,
    scopes: &mut [NoHashMap<u64, ScopeInfo>; 2],
) {
    scopes[0].clear();
    scopes[1].clear();
    let mut depth = 0;
    let id_seed = rng.next();
    let mut ix = 0;
    while ix < events.len() {
        let event = &events[ix];
        match event {
            Event::ScopeStart { id, effect, .. } => {
                // Don't try to overwrite noop events.
                if *effect == Effect::Noop {
                    ix = Tape::find_after_scope_end(events, ix + 1);
                    continue;
                }
                // Ignore trivial scopes (which includes RepeatSize and SelectIndex).
                // Note: this *can* accept the root scope, we'll need to check for it later.
                // Note: we do accept repeat elements here as well.
                // TODO: what we want here is "no primitive scopes" instead, which should include most built-in types
                if events[ix + 1] != Event::ScopeEnd
                    && events[ix + 2] != Event::ScopeEnd
                    && events[ix + 3] != Event::ScopeEnd
                {
                    let d_ix = depth_index(depth);
                    let key = if main { *id ^ id_seed } else { *id };
                    let ent = scopes[d_ix]
                        .entry(key)
                        .or_insert_with(|| ScopeInfo::new(*id));
                    ent.res.accept(rng, ix);
                }
                depth += 1;
            }
            Event::ScopeEnd => {
                depth -= 1;
            }
            _ => {}
        }
        ix += 1;
    }
    debug_assert_eq!(depth, 0);
}

#[cfg(test)]
mod tests {
    use crate::{check, rand::DefaultRand, tape_mutate_crossover::CrossoverCache, tests::RgbState};

    #[test]
    fn crossover_preserves_validity() {
        check(|src| {
            // Draw the things we need *before* `prop_fill_tape(no_err=false)` for tape to remain valid.
            let seed = src.any("seed");
            let t = src.any("temperature");
            let multi = src.any("multi");
            let tape1_as_base = src.any("tape1 as base");
            let tape1_copy_meta = src.any("tape1 copy_meta");
            let tape2_copy_meta = src.any("tape2 copy_meta");

            let mut tape1 = RgbState::default().prop_fill_tape(src, true, false, tape1_copy_meta); // `assume_no_err` first
            let mut tape2 = RgbState::default().prop_fill_tape(src, false, false, tape2_copy_meta); // `!assume_no_err` second
            // Note: tape 2 can contain bigger meta than required (some parts can only be used inside tape 1).
            let (mut base_clone, base, other) = if tape1_as_base {
                (tape1.clone(), &mut tape1, &tape2)
            } else {
                (tape2.clone(), &mut tape2, &tape1)
            };

            // `crossover` contains our debug assertions.
            let mut rng = DefaultRand::new(seed);
            let mut cache = CrossoverCache::default();
            let _ = base.crossover(other, &mut rng, t, multi, true, &mut cache);

            // Ensure that mutation is deterministic.
            let mut rng = DefaultRand::new(seed);
            let mut cache = CrossoverCache::default();
            let _ = base_clone.crossover(other, &mut rng, t, multi, false, &mut cache);
            assert_eq!(&base_clone, base);
        });
    }
}
