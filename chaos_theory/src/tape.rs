// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::sync::Arc;
use core::{
    cmp::Ordering,
    fmt::{self, Debug},
    hash::{Hash as _, Hasher as _},
};
use std::sync::LazyLock;

use crate::{
    Effect, Map, base64,
    hash::FxHasher,
    rand::DefaultRand,
    range::{Range, SizeRange},
    tape_event::{Event, InternId, MetaEvent, ScopeKind},
    tape_mutate::{MutationCache, mutate_events},
    tape_mutate_crossover::{CrossoverCache, crossover_events},
    tape_reduce::TTree,
    tape_validate::Validator,
    varint,
};

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) struct TapeCheckpoint {
    choice_start_ix: u32,
    event_start_ix: u32,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Tape {
    // TODO(meta): remove choices in favor of events without metadata (including min/max)?
    choices: Vec<u64>,  // only stores the "additional" randomness
    events: Vec<Event>, // full history of everything, with metadata
    choice_reuse_ix: u32,
    event_reuse_ix: u32,
    void_reuse_depth: u32,
    next_choice_forced: bool,
    // For mutation, it is more expensive to clear out input metadata
    // (or, especially, to try to generate valid output metadata) than simply
    // to mark the metadata as invalid and ignore it entirely (until we discard it on save).
    meta: Option<TapeMeta>,
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<Option<Tape>>() == 120);

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct TapeMeta {
    interned_lookup: Vec<Arc<str>>,    // quick resolution
    interned: Map<Arc<str>, InternId>, // quick interning
}

impl TapeMeta {
    pub(crate) fn get(&self, id: InternId) -> Option<&Arc<str>> {
        if id.0 == 0 {
            static EMPTY_ARC: LazyLock<Arc<str>> = LazyLock::new(|| String::new().into());
            Some(&EMPTY_ARC)
        } else {
            let ix = id.0 - 1;
            self.interned_lookup.get(ix as usize)
        }
    }

    #[cfg(test)]
    pub(crate) fn intern(&mut self, s: &str) -> (InternId, bool) {
        if s.is_empty() {
            return (InternId(0), false);
        }
        // Can't use the entry API here, since we don't want to construct an Arc prematurely.
        if let Some(id) = self.interned.get(s) {
            debug_assert_ne!(id.0, 0);
            (*id, false)
        } else {
            let id = InternId(self.interned_lookup.len() as u32 + 1);
            let s = Arc::from(s.to_owned());
            self.interned.insert(Arc::clone(&s), id);
            self.interned_lookup.push(s);
            debug_assert_ne!(id.0, 0);
            (id, true)
        }
    }

    fn is_empty(&self) -> bool {
        self.interned_lookup.is_empty() && self.interned.is_empty()
    }

    fn clear(&mut self) {
        self.interned_lookup.clear();
        self.interned.clear();
    }

    fn reserve(&mut self, other: &Self) {
        // TODO(meta): scale like other things in `reserve_for_replay`.
        self.interned_lookup.reserve(other.interned_lookup.len());
        self.interned.reserve(other.interned.len());
    }

    fn rebuild(&mut self, events: &[Event]) {
        debug_assert!(self.is_empty());
        for event in events {
            if let Event::Meta(m) = event {
                match m {
                    MetaEvent::Intern { id, value } => {
                        self.interned_lookup.push(Arc::clone(value));
                        self.interned.insert(Arc::clone(value), *id);
                    }
                }
            }
        }
    }
}

impl fmt::Display for Tape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut depth = 0;
        writeln!(f, "Tape[")?;
        for event in &self.events {
            let empty = "";
            let indent = (depth + 1) * 2;
            if *event != Event::ScopeEnd {
                write!(f, "{empty:>indent$}")?;
                event.format(f, self.meta.as_ref())?;
                writeln!(f)?;
            }
            match event {
                Event::ScopeStart { .. } => depth += 1,
                Event::ScopeEnd => depth -= 1,
                _ => {}
            }
        }
        writeln!(f, "]")
    }
}

impl Tape {
    pub(crate) fn new(meta: bool) -> Self {
        Self {
            meta: meta.then(TapeMeta::default),
            ..Default::default()
        }
    }

    pub(crate) fn from_choices(choices: Vec<u64>) -> Self {
        Self::from_choices_events(choices, Vec::new(), None)
    }

    pub(crate) fn from_events(events: Vec<Event>, build_choices: bool) -> Self {
        let mut choices = Vec::new();
        if build_choices {
            choices.extend(events.iter().filter_map(Event::choice_value));
        }
        Self::from_choices_events(choices, events, None)
    }

    fn from_choices_events(choices: Vec<u64>, events: Vec<Event>, meta: Option<TapeMeta>) -> Self {
        debug_assert!(events.is_empty() || events.len() >= choices.len());
        Self {
            choices,
            events,
            meta,
            ..Default::default()
        }
    }

    pub(crate) fn clear(&mut self) {
        self.choices.clear();
        self.events.clear();
        self.choice_reuse_ix = 0;
        self.event_reuse_ix = 0;
        self.void_reuse_depth = 0;
        self.next_choice_forced = false;
        if let Some(meta) = &mut self.meta {
            meta.clear();
        }
    }

    pub(crate) fn reserve_for_replay(&mut self, replay: &Self) {
        debug_assert!(self.is_empty());
        self.choices.reserve(replay.choices.len() * 2);
        self.events.reserve(replay.events.len() * 2);
        if let (Some(meta), Some(replay_meta)) = (&mut self.meta, &replay.meta) {
            meta.reserve(replay_meta);
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.choices.is_empty()
            && self.events.is_empty()
            && self.meta.as_ref().is_none_or(TapeMeta::is_empty)
    }

    pub(crate) fn has_meta(&self) -> bool {
        self.meta.is_some()
    }

    pub(crate) fn is_void_reuse(&self) -> bool {
        self.void_reuse_depth > 0
    }

    pub(crate) fn reuse_at_zero(&self) -> bool {
        self.choice_reuse_ix == 0 && self.event_reuse_ix == 0 && self.void_reuse_depth == 0
    }

    #[cfg(test)]
    pub(crate) fn as_choices(&self) -> &[u64] {
        &self.choices
    }

    pub(crate) fn smaller_choices(&self, other: &Self) -> bool {
        // Shortlex ordering.
        match self.choices.len().cmp(&other.choices.len()) {
            Ordering::Less => true,
            Ordering::Equal => self.choices < other.choices,
            Ordering::Greater => false,
        }
    }

    pub(crate) fn hash(&self) -> u64 {
        // Use a seed from wyhash, to avoid starting from zero hash which stays zero when adding zeroes to it.
        const SEED: usize = 0x2d358dccaa6c78a5;
        let mut hasher = FxHasher::with_seed(SEED);
        self.events.hash(&mut hasher);
        hasher.finish()
    }

    #[track_caller]
    pub(crate) fn debug_assert_valid(&self) {
        if cfg!(fuzzing) {
            // Don't spend valuable fuzzing time on internal consistency checks
            // (debug assertions in dependencies can be disabled for regular builds
            // but not for cargo-fuzz ones, which sets `RUSTFLAGS -C debug-assertions`).
            return;
        }
        if !cfg!(debug_assertions) {
            // Don't try to validate if we will not assert.
            return;
        }
        if let Err(err) = self.validate() {
            // Use Display impl for prettier output and easier debugging.
            debug_assert_eq!(err, "", "{self}");
        }
    }

    fn validate(&self) -> Result<(), &'static str> {
        if !(0..=self.choices.len()).contains(&(self.choice_reuse_ix as usize)) {
            return Err("choice reuse index out of bounds");
        }
        if !(0..=self.events.len()).contains(&(self.event_reuse_ix as usize)) {
            return Err("event reuse index out of bounds");
        }
        if self.void_reuse_depth != 0 {
            return Err("non-zero void reuse");
        }
        let mut v = Validator::new(self.meta.is_some());
        for event in &self.events {
            v.accept(event)?;
        }
        v.validate(true)?;
        Ok(())
    }

    // Assumes valid tape! `ix` should point to an in-scope event.
    fn find_prev_scope_start(events: &[Event], mut ix: usize) -> usize {
        let mut depth: u32 = 0;
        loop {
            match events.get(ix) {
                Some(Event::ScopeStart { .. }) => {
                    if depth == 0 {
                        return ix;
                    }
                    if ix == 0 {
                        break;
                    }
                    ix -= 1;
                    depth -= 1;
                }
                Some(Event::ScopeEnd) => {
                    if ix == 0 {
                        break;
                    }
                    ix -= 1;
                    depth += 1;
                }
                Some(_) => {
                    if ix == 0 {
                        break;
                    }
                    ix -= 1;
                }
                None => {
                    break;
                }
            }
        }
        unreachable!("internal error: invalid tape; can't find scope start");
    }

    // Assumes valid tape! `ix` should point to an in-scope event.
    pub(crate) fn find_after_scope_end(events: &[Event], mut ix: usize) -> usize {
        let mut depth: u32 = 0;
        loop {
            match events.get(ix) {
                Some(Event::ScopeStart { .. }) => {
                    ix += 1;
                    depth += 1;
                }
                Some(Event::ScopeEnd) => {
                    ix += 1;
                    if depth == 0 {
                        return ix;
                    }
                    depth -= 1;
                }
                Some(_) => {
                    ix += 1;
                }
                None => {
                    unreachable!("internal error: invalid tape; can't find scope end");
                }
            }
        }
    }

    pub(crate) fn find_after_scope_start(
        events: &[Event],
        desired_kind: ScopeKind,
        mut ix: usize,
    ) -> (usize, bool) {
        // Fast-path.
        // Note, we ignore the effect when matching.
        if matches!(events.get(ix), Some(Event::ScopeStart { kind, .. }) if *kind == desired_kind) {
            return (ix + 1, true);
        }
        // If we are not looking for a trailer element but see one, skip them all.
        for trailer_kind in [ScopeKind::RepeatElement, ScopeKind::SelectVariant] {
            // ... and skip any meta events we might find before.
            while matches!(events.get(ix), Some(Event::Meta(..))) {
                ix += 1;
            }
            while desired_kind != trailer_kind
                && matches!(
                    events.get(ix),
                    Some(Event::ScopeStart { kind, .. }) if *kind == trailer_kind
                )
            {
                ix = Self::find_after_scope_end(events, ix + 1);
            }
        }
        // Either consume a scope start of the same kind, or report that we failed to match.
        match events.get(ix) {
            Some(Event::ScopeStart { kind, .. }) if *kind == desired_kind => (ix + 1, true),
            _ => (ix, false),
        }
    }

    pub(crate) fn pop_scope_enter(&mut self, desired_kind: ScopeKind) {
        if self.events.is_empty() {
            return;
        }
        if self.void_reuse_depth > 0 {
            self.void_reuse_depth += 1;
            return;
        }
        let (event_reuse_ix, matched) =
            Self::find_after_scope_start(&self.events, desired_kind, self.event_reuse_ix as usize);
        self.event_reuse_ix = event_reuse_ix as u32;
        if !matched {
            self.void_reuse_depth = 1;
        }
    }

    pub(crate) fn pop_scope_exit(&mut self) {
        if self.events.is_empty() {
            return;
        }
        if self.void_reuse_depth > 0 {
            self.void_reuse_depth -= 1;
            return;
        }
        let event_reuse_ix = Self::find_after_scope_end(&self.events, self.event_reuse_ix as usize);
        self.event_reuse_ix = event_reuse_ix as u32;
    }

    pub(crate) fn pop_choice(&mut self, budget_remaining: &mut usize) -> Option<u64> {
        if self.events.is_empty() {
            if self.choices.is_empty() {
                return None;
            }
            return if let Some(v) = self.choices.get(self.choice_reuse_ix as usize) {
                self.choice_reuse_ix += 1;
                Some(*v)
            } else {
                *budget_remaining = 0;
                None
            };
        }
        if self.void_reuse_depth > 0 {
            return None;
        }
        // Only consume and return an event if it is a choice one
        // (for simplicity we don't care about exact event kind).
        match self.events.get(self.event_reuse_ix as usize) {
            Some(event @ (Event::Size { .. } | Event::Index { .. } | Event::Value { .. })) => {
                self.event_reuse_ix += 1;
                Some(event.unwrap_choice_value())
            }
            Some(_) => None,
            None => {
                *budget_remaining = 0;
                None
            }
        }
    }

    pub(crate) fn push_value(&mut self, v: u64, r: Range<u64>) {
        let e = Event::Value {
            value: v,
            min: r.min,
            max: r.max,
        };
        self.choices.push(e.unwrap_choice_value());
        self.events.push(e);
    }

    pub(crate) fn push_size(&mut self, s: usize, r: SizeRange) {
        let e = Event::Size {
            size: s as u64,
            min: r.min as u64,
            max: r.max as u64,
        };
        self.choices.push(e.unwrap_choice_value());
        self.events.push(e);
    }

    pub(crate) fn push_index(&mut self, ix: usize, n: usize) {
        let e = Event::Index {
            index: ix as u64,
            max: (n - 1) as u64,
            forced: self.next_choice_forced,
        };
        self.choices.push(e.unwrap_choice_value());
        self.events.push(e);
        self.next_choice_forced = false;
    }

    pub(crate) fn mark_next_choice_forced(&mut self) {
        self.next_choice_forced = true;
    }

    pub(crate) fn push_scope_enter(&mut self, scope_id: u64, kind: ScopeKind) {
        self.events.push(Event::ScopeStart {
            id: scope_id,
            kind,
            effect: Effect::Success,
            meta: None, // TODO(meta): fill (conditionally?)
        });
    }

    pub(crate) fn push_scope_exit(&mut self, effect: Effect) {
        if effect != Effect::Success {
            self.scope_mark_unsuccessful(effect);
        }
        self.events.push(Event::ScopeEnd);
    }

    #[inline(never)]
    fn scope_mark_unsuccessful(&mut self, effect: Effect) {
        debug_assert_ne!(effect, Effect::Success);
        let ix = Self::find_prev_scope_start(&self.events, self.events.len() - 1);
        match &mut self.events[ix] {
            Event::ScopeStart {
                kind,
                effect: stored,
                ..
            } if *kind == ScopeKind::RepeatElement && *stored == Effect::Success => {
                *stored = effect;
            }
            _ => unreachable!(
                "internal error: unexpected event where repeat element scope start should be"
            ),
        }
    }

    #[cfg(test)]
    pub(crate) fn checkpoint(&self) -> TapeCheckpoint {
        TapeCheckpoint {
            choice_start_ix: self.choices.len() as u32,
            event_start_ix: self.events.len() as u32,
        }
    }

    #[cfg(test)]
    pub(crate) fn copy_from_checkpoint(
        &self,
        chk: TapeCheckpoint,
        fill_choices: bool,
        copy_meta: bool,
    ) -> Self {
        let choices = if fill_choices {
            self.choices[chk.choice_start_ix as usize..].to_vec()
        } else {
            Vec::new()
        };
        let meta = if copy_meta { self.meta.clone() } else { None };
        Self::from_choices_events(
            choices,
            self.events[chk.event_start_ix as usize..].to_vec(),
            meta,
        )
    }

    pub(crate) fn into_tree(self) -> TTree {
        TTree::from_events(&self.events)
    }

    pub(crate) fn has_noop(&self) -> bool {
        self.events.iter().any(|e| {
            matches!(
                e,
                Event::ScopeStart {
                    effect: Effect::Noop,
                    ..
                }
            )
        })
    }

    pub(crate) fn discard_noop(&self) -> Self {
        // This is just a much faster version of into_tree().to_tape(ignore_noop: true).
        // During reduction, we can spend a lot of time removing noop scopes, so the optimization is worth it.
        let mut choices = Vec::with_capacity(self.choices.len());
        let mut events = Vec::with_capacity(self.events.len());
        let mut ix = 0;
        while ix < self.events.len() {
            let event = self.events[ix].clone();
            match event {
                Event::ScopeStart { effect, .. } => {
                    if effect == Effect::Noop {
                        ix = Self::find_after_scope_end(&self.events, ix + 1);
                        continue;
                    }
                    events.push(event);
                }
                Event::ScopeEnd => {
                    events.push(event);
                }
                Event::Size { .. } | Event::Index { .. } | Event::Value { .. } => {
                    choices.push(event.unwrap_choice_value());
                    events.push(event);
                }
                Event::Meta(..) => {
                    // Discard as well.
                }
            }
            ix += 1;
        }
        Self::from_choices_events(choices, events, None)
    }

    #[expect(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub(crate) fn mutate(
        &mut self,
        rng: &mut DefaultRand,
        t: u8,
        multi: bool,
        shrink_only: bool,
        allow_void: bool,
        debug_validate: bool,
        cache: &mut MutationCache,
    ) {
        debug_assert!(self.reuse_at_zero());
        if debug_validate {
            self.debug_assert_valid();
        }
        self.meta = None;
        mutate_events(
            &mut self.events,
            rng,
            t,
            multi,
            shrink_only,
            allow_void,
            cache,
        );
        self.choices.clear();
        if debug_validate {
            self.debug_assert_valid();
        }
    }

    pub(crate) fn crossover(
        &mut self,
        other: &Self,
        rng: &mut DefaultRand,
        t: u8,
        multi: bool,
        debug_validate: bool,
        cache: &mut CrossoverCache,
    ) -> Self {
        debug_assert!(self.reuse_at_zero());
        debug_assert!(other.reuse_at_zero());
        if debug_validate {
            self.debug_assert_valid();
            other.debug_assert_valid();
        }
        let mut res = Self::default();
        debug_assert!(res.meta.is_none());
        crossover_events(
            &mut self.events,
            &other.events,
            &mut res.events,
            rng,
            t,
            multi,
            cache,
        );
        if debug_validate {
            res.debug_assert_valid();
        }
        res
    }

    // Note: O(N) :-(
    pub(crate) fn events_max_size(&self) -> usize {
        varint::MAX_SIZE + self.events.iter().map(Event::max_size).sum::<usize>()
    }

    // TODO: add header to allow versioning
    pub(crate) fn save_events<'buf>(
        &self,
        mut buf: &'buf mut [u8],
    ) -> Result<&'buf mut [u8], &'static str> {
        debug_assert!(self.reuse_at_zero());
        buf = varint::encode(self.events.len() as u64, buf)?;
        for event in &self.events {
            let ignore_meta = self.meta.is_none();
            buf = event.save(buf, ignore_meta)?;
        }
        Ok(buf)
    }

    const EVENTS_LOAD_LIMIT: usize = 1024 * 1024;

    pub(crate) fn load_events<'buf>(
        &mut self,
        mut buf: &'buf [u8],
        validate: bool,
        build_choices: bool,
    ) -> Result<&'buf [u8], &'static str> {
        debug_assert!(self.is_empty() && self.reuse_at_zero());
        let n;
        (n, buf) = varint::decode(buf)?;
        // Prevent OOM or slow execution when loading events from untrusted input (like fuzzer).
        if n as usize > Self::EVENTS_LOAD_LIMIT {
            return Err("refusing to load more than the maximum number of events");
        }
        self.events.reserve(n as usize);
        for _ in 0..n {
            let event;
            (event, buf) = Event::load(buf, false)?; // we validate all events below
            self.events.push(event);
        }
        if let Some(meta) = &mut self.meta {
            meta.rebuild(&self.events);
        }
        if validate {
            self.validate()?;
        }
        if build_choices {
            self.choices
                .extend(self.events.iter().filter_map(Event::choice_value));
        }
        Ok(buf)
    }

    pub(crate) fn save_events_base64(&self) -> String {
        let mut buf = vec![0; self.events_max_size()];
        let rem_len = self
            .save_events(&mut buf)
            .expect("internal error: failed to save events")
            .len();
        let buf = &buf[..buf.len() - rem_len];
        let mut out = vec![0; base64::encoded_len(buf.len())];
        base64::encode(buf, &mut out).expect("internal error: buffer must be big enough");
        String::from_utf8(out).expect("internal error: base64 data must be valid UTF-8")
    }

    pub(crate) fn load_events_base64(
        &mut self,
        data: &[u8],
        validate: bool,
        build_choices: bool,
    ) -> Result<(), &'static str> {
        let mut buf = vec![0; base64::decoded_len(data.len())];
        let rem = base64::decode(data, &mut buf)?;
        debug_assert!(rem.is_empty());
        let rem = self.load_events(&buf, validate, build_choices)?;
        if !rem.is_empty() {
            return Err("leftover binary data after events");
        }
        Ok(())
    }

    pub(crate) fn save_choices_base64(&self) -> String {
        debug_assert!(self.reuse_at_zero());
        let mut buf = vec![0; self.choices.len() * varint::MAX_SIZE];
        let mut rem = &mut *buf;
        for &u in &self.choices {
            rem = varint::encode(u, rem).expect("internal error: buffer must be big enough");
        }
        let rem_len = rem.len();
        let buf = &buf[..buf.len() - rem_len];
        let mut out = vec![0; base64::encoded_len(buf.len())];
        base64::encode(buf, &mut out).expect("internal error: buffer must be big enough");
        String::from_utf8(out).expect("internal error: base64 data must be valid UTF-8")
    }

    pub(crate) fn load_choices_base64(&mut self, data: &[u8]) -> Result<(), &'static str> {
        debug_assert!(self.is_empty() && self.reuse_at_zero());
        let mut buf = vec![0; base64::decoded_len(data.len())];
        base64::decode(data, &mut buf)?;
        let mut rem = &*buf;
        while !rem.is_empty() {
            let u: u64;
            (u, rem) = varint::decode(rem)?;
            self.choices.push(u);
        }
        // Validate unconditionally, because with only choices, the tape must be always valid.
        self.validate()?;
        Ok(())
    }
}

// TODO: implement Arbitrary for Tape, make it enforce invariants in constructor + put back validation in tests (& mostly kill `prop_fill_tape`)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, tests::RgbState};

    #[test]
    fn tape_events_save_load_roundtrip() {
        check(|src| {
            let fill_choices = src.any("fill_choices");
            let fill_meta = src.any("fill_meta");
            let tape = RgbState::default().prop_fill_tape(src, false, fill_choices, fill_meta);
            let mut buf = vec![0; tape.events_max_size()];
            let rem_len = tape.save_events(&mut buf).unwrap().len();
            let mut tape_ = Tape::new(tape.meta.is_some());
            tape_
                .load_events(&buf[..buf.len() - rem_len], false, fill_choices)
                .unwrap();
            assert_eq!(tape, tape_);
        });
    }

    #[test]
    fn tape_choices_save_load_roundtrip() {
        check(|src| {
            let tape = Tape::from_choices(src.any("choices"));
            let data = tape.save_choices_base64();
            let mut tape_ = Tape::default();
            tape_.load_choices_base64(data.as_bytes()).unwrap();
            assert_eq!(tape, tape_);
        });
    }

    #[test]
    fn tape_tree_roundtrip() {
        check(|src| {
            let tape = RgbState::default().prop_fill_tape(src, false, true, false);
            let tree = tape.clone().into_tree();
            let tape_ = tree.to_tape(false);
            assert_eq!(tape, tape_);
        });
    }

    #[test]
    fn tape_tree_ignore_noop_is_nop() {
        check(|src| {
            let mut state = RgbState::default();
            let tape = state.prop_fill_tape(src, false, true, false);
            let tree = tape.clone().into_tree();
            let tape_disc = tree.to_tape(true);
            if tape.has_noop() {
                assert!(tape_disc.choices.len() < tape.choices.len());
            } else {
                assert_eq!(tape_disc, tape);
            }
            let mut state_ = RgbState::default();
            state_.prop_replay_from_tape(src, tape_disc);
            assert_eq!(state, state_);
        });
    }

    #[test]
    fn tape_discard_noop_is_nop() {
        check(|src| {
            let mut state = RgbState::default();
            let tape = state.prop_fill_tape(src, false, true, false);
            let tape_disc = tape.discard_noop();
            assert!(!tape.has_noop() || tape_disc.choices.len() < tape.choices.len());
            let mut state_ = RgbState::default();
            state_.prop_replay_from_tape(src, tape_disc);
            assert_eq!(state, state_);
        });
    }

    #[test]
    fn tape_meta_intern_roundtrip() {
        check(|src| {
            let mut meta = TapeMeta::default();
            let mut model = Map::<String, InternId>::default();
            let mut model_rev = Map::<InternId, String>::default();
            src.repeat_select(
                "method",
                &["get", "intern"],
                |src, variant, _ix| match variant {
                    "get" => {
                        let mut id_values: Vec<_> = model_rev.iter().collect();
                        id_values.sort();
                        let Some(((id, s), _)) = src.choose("id_value", &id_values) else {
                            return Effect::Noop;
                        };
                        let Some(s_) = meta.get(**id) else {
                            panic!("valid id {id} not found")
                        };
                        assert_eq!(s.as_str(), &**s_);
                        Effect::Success
                    }
                    "intern" => {
                        let s: String = src.any("s");
                        let (id, save_new) = meta.intern(&s);
                        if save_new {
                            let prev = model.insert(s.clone(), id);
                            assert!(prev.is_none());
                            let prev = model_rev.insert(id, s);
                            assert!(prev.is_none());
                        } else if id.0 == 0 {
                            assert!(s.is_empty());
                        } else {
                            let id_ = model.get(&s);
                            assert_eq!(Some(id), id_.copied());
                        }
                        Effect::Success
                    }
                    _ => unreachable!(),
                },
            );
        });
    }
}
