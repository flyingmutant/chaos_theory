// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::mem::take;

use crate::{
    Effect, MAX_SIZE, USE_SEED_AS_IS_PROB, Unsigned as _,
    distrib::{Biased, temperature_scale_down},
    hash_identity::NoHashMap,
    math::percent,
    rand::DefaultRand,
    tape::Tape,
    tape_event::{Event, ScopeKind},
};

// Note: metadata will *not* be valid after mutation.
pub(crate) fn mutate_events(
    events: &mut Vec<Event>,
    rng: &mut DefaultRand,
    t: u8,
    multi: bool,
    shrink_only: bool,
    allow_void: bool,
    cache: &mut MutationCache,
) {
    let n = if multi { mutation_points(rng, t) } else { 1 };
    let (mut n_ok, mut n_nop) = (0, 0);
    while n_ok < n && n_ok + n_nop < MAX_MUTATION_ATTEMPTS {
        let Some((ix, ix2)) = mutation_index(rng, events, shrink_only, allow_void, cache) else {
            break;
        };
        let event = events.get(ix).cloned(); // `.get()` to handle `mutate_scope_void` `usize::MAX` special-case
        let value = match event {
            None | Some(Event::ScopeStart { .. }) => {
                if ix == ix2 {
                    debug_assert!(allow_void);
                    mutate_scope_void(events, ix);
                } else {
                    mutate_scope_donor(events, (ix, ix2), rng, t, shrink_only, allow_void, cache);
                }
                // We consider that we are always successful since we replace scope with a different one.
                n_ok += 1;
                continue;
            }
            Some(Event::ScopeEnd) => unreachable!("internal error: chosen scope end for mutation"),
            Some(Event::Value { value, min, max }) => {
                mutate_value(rng, t, value, min, max, shrink_only)
            }
            Some(Event::Index { index, max, forced }) => {
                debug_assert!(!forced);
                mutate_value(rng, t, index, 0, max, shrink_only)
            }
            Some(Event::Size { size, min, max }) => {
                let is_repeat = matches!(
                    events[ix - 1],
                    Event::ScopeStart {
                        kind: ScopeKind::RepeatSize,
                        ..
                    }
                );
                if is_repeat {
                    mutate_repeat(
                        rng,
                        events,
                        t,
                        ix,
                        min as usize,
                        max as usize,
                        size as usize,
                        shrink_only,
                        allow_void,
                        cache,
                    )
                } else {
                    mutate_value(rng, t, size, min, max, shrink_only)
                        .map(|s| s.min(min + MAX_SIZE as u64))
                }
            }
            Some(Event::Meta(..)) => {
                unreachable!("internal error: chosen meta event for mutation")
            }
        };
        if let Some(value) = value {
            events[ix].set_value(value);
            n_ok += 1;
        } else {
            n_nop += 1;
        }
    }
}

const MAX_MUTATION_POINTS: usize = 8;
const MAX_MUTATION_ATTEMPTS: usize = MAX_MUTATION_POINTS * 4;
const MAX_BIT_FLIP_TRIES: usize = 8;
const MAX_RANDOM_WALK_AMOUNT: usize = 32;
const MUTATE_VALUE_JUMP_TO_BOUND_PROB: f64 = percent(10);
const SELECT_INDEX_MUTATE_PROB: f64 = percent(3);
const REPEAT_INSERT_VOID_PROB: f64 = percent(30);
const MUTATE_ROOT_VOID_PROB: f64 = percent(50);

#[derive(Debug, Default)]
pub(crate) struct MutationCache {
    repeat_scopes: [NoHashMap<u64, Reservoir>; 2],
    select_scopes: [NoHashMap<u64, Reservoir>; 2],
    scope_voids: [NoHashMap<u64, Reservoir>; 2],
    // Note: because we only match by ID, in case of ID collision we can produce an invalid tape.
    scope_pairs: [NoHashMap<u64, Reservoir2>; 2],

    scope_backup: Vec<Event>,
    repeat_backup: Vec<Event>,
    repeat_elements: Vec<Option<(usize, usize)>>,
}

pub(crate) fn rand_or<T>(rng: &mut DefaultRand, lo: Option<T>, hi: Option<T>) -> Option<T> {
    if lo.is_some() && hi.is_some() {
        if rng.coinflip_fair() { lo } else { hi }
    } else {
        lo.or(hi)
    }
}

impl MutationCache {
    fn choose_first_as_random(
        rng: &mut DefaultRand,
        scopes: &[NoHashMap<u64, Reservoir>; 2],
    ) -> Option<usize> {
        let lo = Self::choose_first_as_random_impl(&scopes[0]);
        let hi = Self::choose_first_as_random_impl(&scopes[1]);
        rand_or(rng, lo, hi)
    }

    fn choose_first_pair_as_random(
        rng: &mut DefaultRand,
        scope_pairs: &[NoHashMap<u64, Reservoir2>; 2],
    ) -> Option<(usize, usize)> {
        let lo = Self::choose_first_pair_as_random_impl(&scope_pairs[0]);
        let hi = Self::choose_first_pair_as_random_impl(&scope_pairs[1]);
        rand_or(rng, lo, hi)
    }

    fn choose_first_as_random_impl(scopes: &NoHashMap<u64, Reservoir>) -> Option<usize> {
        // We rely on the fact that we blind all keys with a random seed,
        // so the iteration order should be random and we can just grab the first value.
        scopes.values().next().and_then(|r| r.choice)
    }

    fn choose_first_pair_as_random_impl(
        scope_pairs: &NoHashMap<u64, Reservoir2>,
    ) -> Option<(usize, usize)> {
        // We may have to visit several pairs to find one that is set.
        for r in scope_pairs.values() {
            let c = r.choice();
            if c.is_some() {
                return c;
            }
        }
        None
    }
}

#[derive(Debug, Default, Clone, Copy)]
#[expect(clippy::struct_excessive_bools)]
struct MutationVariants {
    repeat_size: bool,
    select_index: bool,
    free_choice: bool,
    scope_void: bool,
    scope_pair: bool,

    root_void_prob: f64,
}

impl MutationVariants {
    fn actions(self) -> usize {
        usize::from(self.repeat_size)
            + usize::from(self.select_index)
            + usize::from(self.free_choice)
            + usize::from(self.scope_void)
            + usize::from(self.scope_pair)
    }
}

fn mutation_index(
    rng: &mut DefaultRand,
    events: &[Event],
    shrink_only: bool,
    allow_void: bool,
    cache: &mut MutationCache,
) -> Option<(usize, usize)> {
    let mut want = MutationVariants {
        repeat_size: true,
        select_index: rng.coinflip(SELECT_INDEX_MUTATE_PROB),
        free_choice: true,
        scope_void: allow_void,
        scope_pair: true,
        ..Default::default()
    };
    loop {
        let actions = want.actions();
        if actions == 0 {
            return None;
        }
        let mut action = rng.next_below(actions);
        let mut mutation_index_variant =
            |want| mutation_index_variant(rng, events, want, shrink_only, cache);
        if want.repeat_size {
            if action == 0 {
                let r = mutation_index_variant(MutationVariants {
                    repeat_size: true,
                    ..Default::default()
                });
                if r.is_some() {
                    return r;
                }
                want.repeat_size = false;
                continue;
            }
            action -= 1;
        }
        if want.select_index {
            if action == 0 {
                let r = mutation_index_variant(MutationVariants {
                    select_index: true,
                    ..Default::default()
                });
                if r.is_some() {
                    return r;
                }
                want.select_index = false;
                continue;
            }
            action -= 1;
        }
        if want.free_choice {
            if action == 0 {
                let r = mutation_index_variant(MutationVariants {
                    free_choice: true,
                    ..Default::default()
                });
                if r.is_some() {
                    return r;
                }
                want.free_choice = false;
                continue;
            }
            action -= 1;
        }
        if want.scope_void {
            if action == 0 {
                let r = mutation_index_variant(MutationVariants {
                    scope_void: true,
                    root_void_prob: MUTATE_ROOT_VOID_PROB,
                    ..Default::default()
                });
                if r.is_some() {
                    return r;
                }
                want.scope_void = false;
                continue;
            }
            action -= 1;
        }
        if want.scope_pair {
            debug_assert_eq!(action, 0);
            let r = mutation_index_variant(MutationVariants {
                scope_pair: true,
                ..Default::default()
            });
            if r.is_some() {
                return r;
            }
            want.scope_pair = false;
            continue;
        }
        unreachable!("internal error: wrong mutate index action choice logic");
    }
}

#[derive(Debug, Default)]
pub(crate) struct Reservoir {
    pub(crate) choice: Option<usize>,
    seen: usize,
}

impl Reservoir {
    pub(crate) fn accept(&mut self, rng: &mut DefaultRand, ix: usize) {
        if self.choice.is_none() || rng.next_below(self.seen + 1) == 0 {
            self.choice = Some(ix);
        }
        self.seen += 1;
    }
}

#[derive(Debug, Default)]
struct Reservoir2 {
    choices: (Option<usize>, Option<usize>),
    seen: usize,
}

impl Reservoir2 {
    fn accept(&mut self, rng: &mut DefaultRand, ix: usize) {
        if self.choices.0.is_none() {
            self.choices.0 = Some(ix);
        } else if self.choices.1.is_none() {
            self.choices.1 = Some(ix);
        } else {
            match rng.next_below(self.seen + 1) {
                0 => self.choices.0 = Some(ix),
                1 => self.choices.1 = Some(ix),
                _ => {}
            }
        }
        self.seen += 1;
    }

    fn choice(&self) -> Option<(usize, usize)> {
        if let (Some(a), Some(b)) = self.choices {
            Some((a, b))
        } else {
            None
        }
    }
}

// Low depth -> 0, high depth -> 1.
// Note: this does not play nicely with `Source::scope()` or other things that inflate the depth :-(
pub(crate) fn depth_index(depth_scope_start: usize) -> usize {
    usize::from(depth_scope_start >= 2)
}

fn depth_index_choice(depth_choice: usize) -> usize {
    // Handle choices at the top (bare, without scopes) properly.
    depth_index(depth_choice.saturating_sub(1))
}

#[expect(clippy::too_many_lines)]
fn mutation_index_variant(
    rng: &mut DefaultRand,
    events: &[Event],
    want: MutationVariants,
    shrink_only: bool,
    cache: &mut MutationCache,
) -> Option<(usize, usize)> {
    if want.scope_void && rng.coinflip(want.root_void_prob) {
        return Some((usize::MAX, usize::MAX));
    }
    for i in [0, 1] {
        cache.repeat_scopes[i].clear();
        cache.select_scopes[i].clear();
        cache.scope_voids[i].clear();
        cache.scope_pairs[i].clear();
    }
    let id_seed = rng.next(); // randomize the key iteration order
    let mut next_repeat_size = None;
    let mut next_select_index = None;
    // What is the reason free choice is the only thing we don't group by scope ID first?
    let mut free = [Reservoir::default(), Reservoir::default()];
    let mut depth = 0;
    let mut ix = 0;
    while ix < events.len() {
        let e = &events[ix];
        match e {
            Event::ScopeStart {
                id,
                kind,
                effect,
                meta: _,
            } => {
                // Don't try to mutate noop events.
                if *effect == Effect::Noop {
                    ix = Tape::find_after_scope_end(events, ix + 1);
                    continue;
                }
                let scope_key = id ^ id_seed;
                next_repeat_size = (*kind == ScopeKind::RepeatSize).then_some(scope_key);
                next_select_index = (*kind == ScopeKind::SelectIndex).then_some(scope_key);
                if want.scope_void || want.scope_pair {
                    // Avoid trivial scopes.
                    // TODO: what we want here is "no primitive scopes" instead, which should include most built-in types
                    if events[ix + 1] != Event::ScopeEnd
                        && events[ix + 2] != Event::ScopeEnd
                        && events[ix + 3] != Event::ScopeEnd
                    {
                        let d_ix = depth_index(depth);
                        if want.scope_void {
                            // Scope can possibly be an explicit root one.
                            let r = cache.scope_voids[d_ix].entry(scope_key).or_default();
                            r.accept(rng, ix);
                        }
                        if want.scope_pair {
                            // No need to check for root scope as it will not form a pair.
                            let r = cache.scope_pairs[d_ix].entry(scope_key).or_default();
                            r.accept(rng, ix);
                        }
                    }
                }
                depth += 1;
            }
            Event::ScopeEnd => {
                depth -= 1;
            }
            Event::Size { size, min, max } => {
                let can_change = if shrink_only { size > min } else { min != max };
                let d_ix = depth_index_choice(depth);
                if let Some(scope_key) = next_repeat_size {
                    next_repeat_size = None;
                    if want.repeat_size && (can_change || (!shrink_only && *size > 1)) {
                        let r = cache.repeat_scopes[d_ix].entry(scope_key).or_default();
                        r.accept(rng, ix);
                    }
                } else if want.free_choice && can_change {
                    free[d_ix].accept(rng, ix);
                }
            }
            Event::Index { index, max, forced } => {
                let can_change = (if shrink_only { *index > 0 } else { *max != 0 }) && !forced;
                let d_ix = depth_index_choice(depth);
                if let Some(scope_key) = next_select_index {
                    next_select_index = None;
                    if want.select_index && can_change {
                        let r = cache.select_scopes[d_ix].entry(scope_key).or_default();
                        r.accept(rng, ix);
                    }
                } else if want.free_choice && can_change {
                    free[d_ix].accept(rng, ix);
                }
            }
            Event::Value {
                value, min, max, ..
            } => {
                let can_change = if shrink_only { value > min } else { min != max };
                if want.free_choice && can_change {
                    free[depth_index_choice(depth)].accept(rng, ix);
                }
            }
            Event::Meta(..) => {}
        }
        ix += 1;
    }
    debug_assert_eq!(depth, 0);
    if want.repeat_size {
        return MutationCache::choose_first_as_random(rng, &cache.repeat_scopes).map(|i| (i, 0));
    }
    if want.select_index {
        return MutationCache::choose_first_as_random(rng, &cache.select_scopes).map(|i| (i, 0));
    }
    if want.free_choice {
        return rand_or(rng, free[0].choice, free[1].choice).map(|i| (i, 0));
    }
    if want.scope_void {
        return MutationCache::choose_first_as_random(rng, &cache.scope_voids).map(|i| (i, i));
    }
    if want.scope_pair {
        return MutationCache::choose_first_pair_as_random(rng, &cache.scope_pairs);
    }
    unreachable!("internal error: no mutation variant to want");
}

fn mutate_scope_void(events: &mut Vec<Event>, ix: usize) {
    if ix == usize::MAX {
        events.clear();
    } else {
        debug_assert!(matches!(events[ix], Event::ScopeStart { .. }));
        let ix_end = Tape::find_after_scope_end(events, ix + 1);
        events.drain(ix + 1..ix_end - 1);
    }
}

fn mutate_scope_donor(
    events: &mut Vec<Event>,
    ix: (usize, usize),
    rng: &mut DefaultRand,
    t: u8,
    shrink_only: bool,
    allow_void: bool,
    cache: &mut MutationCache,
) {
    debug_assert_ne!(ix.0, ix.1);
    debug_assert!(
        matches!((&events[ix.0], &events[ix.1]), (&Event::ScopeStart { id: id0, .. }, &Event::ScopeStart { id: id1, .. }) if id0 == id1)
    );
    let ix0_end = Tape::find_after_scope_end(events, ix.0 + 1);
    let ix1_end = Tape::find_after_scope_end(events, ix.1 + 1);
    // We need to remove it from the cache so that we avoid reentrancy.
    let mut donor_scope = take(&mut cache.scope_backup);
    donor_scope.clear();
    // Operate only on events *between* scope start/end. That way even if we erase all of them, valid start/end pair will remain.
    donor_scope.extend_from_slice(&events[ix.1 + 1..ix1_end - 1]);
    if !rng.coinflip(USE_SEED_AS_IS_PROB) {
        mutate_events(
            &mut donor_scope,
            rng,
            t,
            true,
            shrink_only,
            allow_void,
            cache,
        );
    }
    events.splice(ix.0 + 1..ix0_end - 1, donor_scope.iter().cloned());
    cache.scope_backup = donor_scope;
}

pub(crate) fn mutation_points(rng: &mut DefaultRand, t: u8) -> usize {
    // Low temperature to have a few mutation points.
    let dist = Biased::new_temperature(temperature_scale_down(t), None);
    let n = dist.sample(rng, MAX_MUTATION_POINTS);
    n + 1
}

fn bit_to_flip(rng: &mut DefaultRand, t: u8, bits: usize) -> usize {
    let dist = Biased::new_temperature(t, None);
    dist.sample(rng, bits)
}

fn random_delta_amount(rng: &mut DefaultRand, t: u8, max: u64, jump: bool) -> u64 {
    if max == 0 {
        0
    } else if jump {
        // Uniform jump. Should we do a non-uniform draw instead, like in `choice_new_value`?
        rng.next_below_u64(max) + 1
    } else {
        let dist = Biased::new_temperature(t, None);
        let n = dist.sample(rng, (max as usize).min(MAX_RANDOM_WALK_AMOUNT));
        n as u64 + 1
    }
}

fn repeat_modify_amount(rng: &mut DefaultRand, t: u8, mut max_size: usize) -> usize {
    debug_assert_ne!(max_size, 0);
    max_size = max_size.min(MAX_SIZE);
    // Low temperature to modify usually only a few elements.
    let dist = Biased::new_temperature(temperature_scale_down(t), None);
    let n = dist.sample(rng, max_size);
    n + 1
}

fn mutate_value(
    rng: &mut DefaultRand,
    t: u8,
    value: u64,
    min: u64,
    max: u64,
    shrink_only: bool,
) -> Option<u64> {
    debug_assert_ne!(min, max);
    if shrink_only {
        debug_assert_ne!(value, min);
    }
    let v = match rng.next_below(3) {
        0 => mutate_value_bitflip(rng, t, value, min, max, shrink_only)?,
        1 => mutate_value_delta(rng, t, value, min, max, shrink_only, false),
        2 => mutate_value_delta(rng, t, value, min, max, shrink_only, true),
        _ => unreachable!("internal error: impossible random choice"),
    };
    debug_assert_ne!(v, value);
    debug_assert!((min..=max).contains(&v));
    Some(v)
}

fn mutate_value_bitflip(
    rng: &mut DefaultRand,
    t: u8,
    value: u64,
    min: u64,
    max: u64,
    shrink_only: bool,
) -> Option<u64> {
    let bits = max.bit_len();
    debug_assert_ne!(bits, 0);
    // There is a chance we will not hit inside our range, so loop a bit.
    for _ in 0..MAX_BIT_FLIP_TRIES {
        let mut bit = bit_to_flip(rng, t, bits);
        if shrink_only && value & (1 << bit) == 0 {
            // Slow path, choose only between set bits.
            let mut set_bits = [0; u64::BITS as usize];
            let mut n = 0;
            for i in 0..bits {
                if value & (1 << i) != 0 {
                    set_bits[n] = i;
                    n += 1;
                }
            }
            debug_assert_ne!(n, 0);
            bit = set_bits[bit_to_flip(rng, t, n)];
            debug_assert_ne!(value & (1 << bit), 0);
        }
        let v = value ^ (1 << bit);
        if (min..=max).contains(&v) {
            return Some(v);
        }
    }
    None
}

fn mutate_value_delta(
    rng: &mut DefaultRand,
    t: u8,
    value: u64,
    min: u64,
    max: u64,
    shrink_only: bool,
    jump: bool,
) -> u64 {
    let can_shrink = value > min;
    let can_grow = !shrink_only && value < max;
    let actions = usize::from(can_shrink) + usize::from(can_grow);
    let mut action = rng.next_below(actions);
    if can_shrink {
        if action == 0 {
            if jump && rng.coinflip(MUTATE_VALUE_JUMP_TO_BOUND_PROB) {
                return min;
            }
            return value - random_delta_amount(rng, t, value - min, jump);
        }
        action -= 1;
    }
    if can_grow {
        debug_assert_eq!(action, 0);
        if jump && rng.coinflip(MUTATE_VALUE_JUMP_TO_BOUND_PROB) {
            return max;
        }
        return value + random_delta_amount(rng, t, max - value, jump);
    }
    unreachable!("internal error: wrong mutate action choice logic");
}

#[expect(clippy::too_many_arguments)]
fn mutate_repeat(
    rng: &mut DefaultRand,
    events: &mut Vec<Event>,
    t: u8,
    size_ix: usize,
    min: usize,
    max: usize,
    count: usize,
    shrink_only: bool,
    allow_void: bool,
    cache: &mut MutationCache,
) -> Option<u64> {
    static VOID_REPEAT_ELEMENT: &[Event] = &[
        Event::ScopeStart {
            id: u64::MAX,
            kind: ScopeKind::RepeatElement,
            effect: Effect::Success,
            meta: None,
        },
        Event::ScopeEnd,
    ];

    let elements = &mut cache.repeat_elements;
    elements.clear();
    elements.reserve(count);
    let (begin, end) = grab_repeat(events, size_ix, elements);
    let new_count = mutate_repeat_elements(rng, t, min, max, elements, shrink_only, allow_void)?;
    let backup = &mut cache.repeat_backup;
    backup.clear();
    backup.extend_from_slice(&events[begin..end]);
    events.splice(
        begin..end,
        elements
            .iter()
            .flat_map(|e| {
                if let &Some((i, j)) = e {
                    &backup[i - begin..j - begin]
                } else {
                    VOID_REPEAT_ELEMENT
                }
            })
            .cloned(),
    );
    if (min..=max).contains(&(new_count as usize)) {
        Some(new_count)
    } else {
        // We were dealing with an incomplete tape, and have successfully
        // mutated it. However, it is still an incomplete one. Let's leave
        // the old count in place, as an aspirational target (we can't use
        // the new count, as it is outside the valid range).
        Some(count as u64)
    }
}

fn grab_repeat(
    events: &[Event],
    size_ix: usize,
    elements: &mut Vec<Option<(usize, usize)>>,
) -> (usize, usize) {
    debug_assert!(matches!(events[size_ix], Event::Size { .. }));
    debug_assert!(matches!(events[size_ix + 1], Event::ScopeEnd));
    let begin = size_ix + 2;
    let mut end = begin;
    loop {
        let event = events.get(end);
        match event {
            Some(Event::ScopeStart { kind, effect, .. }) => {
                if *kind != ScopeKind::RepeatElement {
                    break;
                }
                let i = Tape::find_after_scope_end(events, end + 1);
                // Skip all noop elements; hopefully they all were truly no-op.
                if *effect != Effect::Noop {
                    elements.push(Some((end, i)));
                }
                end = i;
            }
            _ => break,
        }
    }
    (begin, end)
}

fn mutate_repeat_elements(
    rng: &mut DefaultRand,
    t: u8,
    min: usize,
    max: usize,
    elements: &mut Vec<Option<(usize, usize)>>,
    shrink_only: bool,
    allow_void: bool,
) -> Option<u64> {
    // Note: the repeat might be from an incomplete tape (elements.len() < min).
    // elements.len() > max is also trivially possible (with Effect::Change steps).
    for _ in 0..mutation_points(rng, t) {
        let can_shrink = elements.len() > min;
        let can_reorder = !shrink_only && elements.len() > 1;
        let can_grow = !shrink_only && elements.len() < max && (allow_void || !elements.is_empty());
        let actions = usize::from(can_shrink) + usize::from(can_reorder) + usize::from(can_grow);
        if actions == 0 {
            // This will return on the first iteration or never.
            return None;
        }
        let mut action = rng.next_below(actions);
        if can_shrink {
            if action == 0 {
                // Delete elements.
                let n = repeat_modify_amount(rng, t, elements.len() - min);
                let from = rng.next_below(elements.len() - n + 1);
                elements.drain(from..from + n);
                continue;
            }
            action -= 1;
        }
        if can_reorder {
            if action == 0 {
                // Move elements.
                // TODO: maybe do partial shuffle instead?
                let n = repeat_modify_amount(rng, t, elements.len() / 2);
                let from = rng.next_below(elements.len() - n + 1);
                let cut = elements.drain(from..from + n).collect::<Vec<_>>();
                let pos = rng.next_below(elements.len() + 1);
                elements.splice(pos..pos, cut);
                continue;
            }
            action -= 1;
        }
        if can_grow {
            debug_assert_eq!(action, 0);
            // Insert elements.
            let n = repeat_modify_amount(rng, t, max - elements.len());
            let from = rng.next_below(elements.len() + 1);
            if allow_void && (elements.is_empty() || rng.coinflip(REPEAT_INSERT_VOID_PROB)) {
                elements.splice(from..from, core::iter::repeat_n(None, n));
            } else {
                let mut ins = Vec::with_capacity(n);
                let mut i = rng.next_below(elements.len());
                for _ in 0..n {
                    ins.push(elements[i]);
                    i = if i == elements.len() - 1 { 0 } else { i + 1 };
                }
                elements.splice(from..from, ins);
            }
            continue;
        }
        unreachable!("internal error: wrong repeat mutate action choice logic");
    }
    Some(elements.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::MutationCache;
    use crate::{check, rand::DefaultRand, tape::Tape, tests::RgbState};

    #[test]
    fn empty_tape_mutate_nop() {
        // This test can be removed once we replace RgbState random tape below with a truly random one (that can sometimes be empty).
        check(|src| {
            let with_meta = src.any("with_meta");
            let mut tape = Tape::new(with_meta);
            let seed = src.any("seed");
            let t = src.any("temperature");
            let multi = src.any("multi");
            let shrink_only = src.any("shrink_only");
            let allow_void = src.any("allow_void");

            let mut rng = DefaultRand::new(seed);
            let mut cache = MutationCache::default();
            tape.mutate(
                &mut rng,
                t,
                multi,
                shrink_only,
                allow_void,
                true,
                &mut cache,
            );
            assert_eq!(tape, Tape::default());
        });
    }

    #[test]
    fn mutate_preserves_validity() {
        check(|src| {
            // Draw the things we need *before* `prop_fill_tape(no_err=false)` for tape to remain valid.
            let copy_meta = src.any("copy_meta");
            let seed = src.any("seed");
            let t = src.any("temperature");
            let multi = src.any("multi");
            let shrink_only = src.any("shrink_only");
            let allow_void = src.any("allow_void");

            let mut tape = RgbState::default().prop_fill_tape(src, false, false, copy_meta);
            let mut tape_clone = tape.clone();

            // `mutate` contains our debug assertions.
            let mut rng = DefaultRand::new(seed);
            let mut cache = MutationCache::default();
            tape.mutate(
                &mut rng,
                t,
                multi,
                shrink_only,
                allow_void,
                true,
                &mut cache,
            );

            // Ensure that mutation is deterministic.
            let mut rng = DefaultRand::new(seed);
            let mut cache = MutationCache::default();
            tape_clone.mutate(
                &mut rng,
                t,
                multi,
                shrink_only,
                allow_void,
                false,
                &mut cache,
            );
            assert_eq!(tape_clone, tape);
        });
    }
}
