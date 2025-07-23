// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// TODO(meta): replace all this with new meta-events

use crate::tape_event::ScopeKind;

#[derive(Debug)]
pub(crate) struct Cover {
    _depth: usize,
    require: bool,
}

impl Cover {
    pub(crate) fn new(depth: usize, require: bool) -> Self {
        debug_assert!(depth > 0 || require);
        Self {
            _depth: depth,
            require,
        }
    }

    pub(crate) fn require(&self) -> bool {
        self.require
    }

    #[expect(clippy::unused_self)]
    pub(crate) fn done(&self) -> bool {
        // TODO(meta)
        true
    }

    #[inline(never)]
    #[expect(clippy::unused_self)]
    pub(crate) fn on_scope_enter(
        &mut self,
        _label: &str,
        _variant: &str,
        _kind: ScopeKind,
        _counter: Option<u32>,
    ) {
        // TODO(meta)
    }

    #[inline(never)]
    #[expect(clippy::unused_self)]
    pub(crate) fn on_scope_exit(&mut self) {
        // TODO(meta)
    }

    #[inline(never)]
    #[expect(clippy::unused_self)]
    pub(crate) fn cover_all(&mut self, _conditions: &[(&str, bool)]) {
        // TODO(meta)
    }

    #[inline(never)]
    #[expect(clippy::unused_self)]
    pub(crate) fn cover_any(&mut self, _conditions: &[(&str, bool)]) {
        // TODO(meta)
    }
}
