// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::sync::Arc;
use core::num::NonZero;

use crate::{
    Effect, Set,
    tape_event::{Event, InternId, MetaEvent, ScopeKind},
};

pub(crate) struct Validator {
    root_state: ValidatorState,
    scopes: Vec<(ValidatorState, Effect)>,
    metadata_valid: bool,
    next_intern_id: InternId,
    interned: Set<Arc<str>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidatorState {
    Default,
    AfterRepeatSizeStart,
    AfterRepeatSizeChoice(usize),
    InRepeat(NonZero<usize>),
    AfterSelectIndexStart,
    AfterSelectIndexChoice,
    InSelect,
    EarlyEnd,
}

impl Validator {
    pub(crate) fn new(metadata_valid: bool) -> Self {
        Self {
            root_state: ValidatorState::Default,
            scopes: Vec::new(),
            metadata_valid,
            next_intern_id: InternId(1),
            interned: Set::default(),
        }
    }

    fn state(&self) -> ValidatorState {
        self.scopes.last().map_or(self.root_state, |s| s.0)
    }

    fn set_state(&mut self, state: ValidatorState) {
        *self
            .scopes
            .last_mut()
            .map(|s| &mut s.0)
            .unwrap_or(&mut self.root_state) = state;
    }

    pub(crate) fn accept(&mut self, event: &Event) -> Result<(), &'static str> {
        type VS = ValidatorState;
        event.validate()?;
        let state = self.state();
        match event {
            Event::ScopeStart {
                id: _,
                kind,
                effect,
                meta,
            } => {
                if let Some(meta) = meta
                    && self.metadata_valid
                {
                    if meta.label >= self.next_intern_id {
                        return Err("invalid label intern ID");
                    }
                    if meta.variant >= self.next_intern_id {
                        return Err("invalid variant intern ID");
                    }
                }
                let new_scope_state = Self::accept_scope_start(state, *kind, *effect)?;
                self.scopes.push((new_scope_state, *effect));
            }
            Event::ScopeEnd => {
                let (child_state, effect) = self.scopes.pop().ok_or("unbalanced scope end")?;
                let parent_state = self.state();
                let new_parent_state = Self::accept_scope_end(child_state, parent_state, effect)?;
                if let Some(state) = new_parent_state {
                    self.set_state(state);
                }
            }
            Event::Size { size, .. } => match state {
                VS::Default => {}
                VS::AfterRepeatSizeStart => {
                    self.set_state(VS::AfterRepeatSizeChoice(*size as usize));
                }
                _ => return Err("size choice in unexpected state"),
            },
            Event::Index { .. } => match state {
                VS::Default => {}
                VS::AfterSelectIndexStart => {
                    self.set_state(VS::AfterSelectIndexChoice);
                }
                _ => return Err("index choice in unexpected state"),
            },
            Event::Value { .. } => match state {
                VS::Default => {}
                _ => return Err("value choice in unexpected state"),
            },
            Event::Meta(meta) => {
                // TODO(meta): check that meta events only occur at the right points relative to others; we use this property to skip them properly (e.g. we don't expect them between repeat elements)
                if self.metadata_valid {
                    match meta {
                        MetaEvent::Intern { id, value } => {
                            if *id != self.next_intern_id {
                                return Err("non-consecutive interned value ID");
                            }
                            if !self.interned.insert(Arc::clone(value)) {
                                return Err("duplicate intern of a value");
                            }
                            self.next_intern_id = InternId(self.next_intern_id.0 + 1);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn accept_scope_start(
        state: ValidatorState,
        kind: ScopeKind,
        effect: Effect,
    ) -> Result<ValidatorState, &'static str> {
        type VS = ValidatorState;
        type SK = ScopeKind;
        // Repeat (e.g. from `find`) can end early (without success),
        // so in most cases we treat it the same as default.
        match kind {
            SK::Plain => match state {
                VS::Default | VS::InRepeat(_) => Ok(VS::Default),
                _ => Err("scope start in unexpected state"),
            },
            SK::RepeatSize => match state {
                VS::Default | VS::InRepeat(_) => Ok(VS::AfterRepeatSizeStart),
                _ => Err("repeat size scope start in unexpected state"),
            },
            SK::RepeatElement => match state {
                VS::InRepeat(_) => Ok(VS::Default),
                _ => {
                    if effect == Effect::Noop {
                        // During reduction, some noop tape elements can appear
                        // after the required number of normal elements; ignore them.
                        Ok(VS::Default)
                    } else {
                        Err("repeat element not immediately after repeat size or another element")
                    }
                }
            },
            SK::SelectIndex => match state {
                VS::Default | VS::InRepeat(_) => Ok(VS::AfterSelectIndexStart),
                _ => Err("select index scope start in unexpected state"),
            },
            SK::SelectVariant => match state {
                VS::InSelect => Ok(VS::Default),
                _ => Err("select variant not immediately after select index"),
            },
        }
    }

    fn accept_scope_end(
        child_state: ValidatorState,
        parent_state: ValidatorState,
        effect: Effect,
    ) -> Result<Option<ValidatorState>, &'static str> {
        type VS = ValidatorState;
        match child_state {
            VS::Default => match parent_state {
                VS::InRepeat(n) => match effect {
                    Effect::Noop | Effect::Change => Ok(None),
                    Effect::Success => {
                        if let Ok(m) = (n.get() - 1).try_into() {
                            Ok(Some(VS::InRepeat(m)))
                        } else {
                            Ok(Some(VS::Default))
                        }
                    }
                },
                VS::InSelect => Ok(Some(VS::Default)),
                VS::Default => Ok(None),
                VS::AfterRepeatSizeStart
                | VS::AfterRepeatSizeChoice(_)
                | VS::AfterSelectIndexStart
                | VS::AfterSelectIndexChoice
                | VS::EarlyEnd => {
                    Err("internal validation error: unexpected validator state at scope end")
                }
            },
            VS::AfterRepeatSizeStart => Err("missing repeat size"),
            VS::AfterRepeatSizeChoice(n) => {
                if let Ok(n) = n.try_into() {
                    Ok(Some(VS::InRepeat(n)))
                } else {
                    Ok(Some(VS::Default))
                }
            }
            VS::AfterSelectIndexStart => Err("missing select index"),
            VS::AfterSelectIndexChoice => Ok(Some(VS::InSelect)),
            VS::InRepeat(_) => Ok(None), // e.g. unsuccessful find
            VS::InSelect | VS::EarlyEnd => Ok(Some(VS::EarlyEnd)),
        }
    }

    // Tapes that resulted in panics can end early.
    pub(crate) fn validate(&self, accept_early_end: bool) -> Result<(), &'static str> {
        type VS = ValidatorState;
        if !self.scopes.is_empty() {
            return Err("unbalanced scope start");
        }
        match self.root_state {
            VS::Default => Ok(()),
            VS::AfterRepeatSizeStart
            | VS::AfterRepeatSizeChoice(_)
            | VS::AfterSelectIndexStart
            | VS::AfterSelectIndexChoice => {
                Err("internal validation error: unexpected validator state at root")
            }
            VS::InRepeat(_) | VS::InSelect | VS::EarlyEnd => {
                if accept_early_end {
                    Ok(())
                } else {
                    Err("unexpected early end of tape")
                }
            }
        }
    }
}
