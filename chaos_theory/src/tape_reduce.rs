// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::{
    Effect,
    reduce::{Seq, Tree, TreeNodeChild, reduce_num, reduce_seq},
    tape::Tape,
    tape_event::{Event, ScopeKind},
};

#[derive(Clone, Debug)]
struct TTreeNode {
    scope_id: u64,
    scope_kind: ScopeKind,
    scope_effect: Effect,
    parent_id: Option<usize>,
    children: Vec<TTreeChild>,
}

impl TTreeNode {
    fn new(
        parent_id: Option<usize>,
        scope_id: u64,
        scope_kind: ScopeKind,
        scope_effect: Effect,
    ) -> Self {
        Self {
            scope_id,
            scope_kind,
            scope_effect,
            parent_id,
            children: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum TTreeChild {
    Choice(Event),
    Scope {
        id: usize,
    },
    Repeat {
        id: usize,
        size: Option<Event>,
        elements: Vec<Option<(usize, Effect)>>,
    },
}

// Tape tree
pub(crate) struct TTree {
    nodes: Vec<TTreeNode>,
}

impl TTree {
    pub(crate) fn from_events(events: &[Event]) -> Self {
        let mut tree = Self { nodes: Vec::new() };
        tree.add_events(events);
        tree
    }

    fn add_node(&mut self, node: TTreeNode) -> usize {
        let id = self.nodes.len();
        if let Some(parent_id) = node.parent_id {
            let parent = &mut self.nodes[parent_id];
            match node.scope_kind {
                ScopeKind::RepeatSize => {
                    parent.children.push(TTreeChild::Repeat {
                        id,
                        size: None,
                        elements: Vec::new(),
                    });
                }
                ScopeKind::RepeatElement => {
                    let TTreeChild::Repeat { elements, .. } = parent
                        .children
                        .last_mut()
                        .expect("internal error: missing child for repeat")
                    else {
                        unreachable!("internal error: repeat element following normal node");
                    };
                    elements.push(Some((id, node.scope_effect)));
                }
                _ => {
                    parent.children.push(TTreeChild::Scope { id });
                }
            }
        }
        self.nodes.push(node);
        id
    }

    fn node(&mut self, id: Option<usize>) -> &TTreeNode {
        &self.nodes[id.expect("internal error: node id must be set")]
    }

    fn node_mut(&mut self, id: Option<usize>) -> &mut TTreeNode {
        &mut self.nodes[id.expect("internal error: node id must be set")]
    }

    fn add_events(&mut self, events: &[Event]) {
        let root = TTreeNode::new(None, 0, ScopeKind::Plain, Effect::Success);
        let mut cur_node_id = Some(self.add_node(root));
        debug_assert_eq!(cur_node_id, Some(0));
        let mut fixup_repeat_size = false;
        for event in events {
            match event {
                Event::ScopeStart {
                    id,
                    kind,
                    effect,
                    meta: _,
                } => {
                    let node = TTreeNode::new(cur_node_id, *id, *kind, *effect);
                    cur_node_id = Some(self.add_node(node));
                    fixup_repeat_size = *kind == ScopeKind::RepeatSize;
                }
                Event::ScopeEnd => {
                    let node = self.node_mut(cur_node_id);
                    debug_assert!(node.parent_id.is_some());
                    cur_node_id = node.parent_id;
                }
                Event::Size { .. } | Event::Index { .. } | Event::Value { .. } => {
                    if fixup_repeat_size {
                        fixup_repeat_size = false;
                        let parent_id = self.node(cur_node_id).parent_id;
                        let TTreeChild::Repeat { size, .. } = self
                            .node_mut(parent_id)
                            .children
                            .last_mut()
                            .expect("internal error: missing child for repeat")
                        else {
                            unreachable!("internal error: repeat element following normal node");
                        };
                        debug_assert!(size.is_none());
                        *size = Some(event.clone());
                    }
                    let node = self.node_mut(cur_node_id);
                    node.children.push(TTreeChild::Choice(event.clone()));
                }
                Event::Meta(..) => {}
            }
        }
        debug_assert_eq!(cur_node_id, Some(0));
    }

    pub(crate) fn to_tape(&self, ignore_noop: bool) -> Tape {
        debug_assert!(!self.nodes.is_empty());
        let mut events = Vec::new();
        self.to_tape_rec(&mut events, 0, ignore_noop);
        Tape::from_events(events, true)
    }

    fn to_tape_rec(&self, events: &mut Vec<Event>, id: usize, ignore_noop: bool) {
        let node = &self.nodes[id];
        if node.scope_id != 0 {
            events.push(Event::ScopeStart {
                id: node.scope_id,
                kind: node.scope_kind,
                effect: node.scope_effect,
                meta: None,
            });
        }
        for child in &node.children {
            match child {
                TTreeChild::Choice(event) => events.push(event.clone()),
                TTreeChild::Scope { id } => self.to_tape_rec(events, *id, ignore_noop),
                TTreeChild::Repeat { id, elements, .. } => {
                    self.to_tape_rec(events, *id, ignore_noop);
                    for (id, effect) in elements.iter().flatten() {
                        if !ignore_noop || *effect != Effect::Noop {
                            self.to_tape_rec(events, *id, ignore_noop);
                        }
                    }
                }
            }
        }
        if node.scope_id != 0 {
            events.push(Event::ScopeEnd);
        }
    }
}

impl Tree for TTree {
    type NodeId = usize;
    type Child = TTreeChild;

    fn root(&self) -> Option<Self::NodeId> {
        (!self.nodes.is_empty()).then_some(0)
    }

    fn children_num(&self, node_id: Self::NodeId) -> usize {
        let node = &self.nodes[node_id];
        node.children.len()
    }

    fn child(&self, node_id: Self::NodeId, ix: usize) -> Self::Child {
        let node = &self.nodes[node_id];
        node.children[ix].clone()
    }

    fn child_replace(&mut self, node_id: Self::NodeId, ix: usize, child: &Self::Child) {
        if let TTreeChild::Repeat { id, size, .. } = child {
            self.nodes[*id].children = vec![TTreeChild::Choice(
                size.clone().expect("internal error: unset repeat size"),
            )];
        }
        let node = &mut self.nodes[node_id];
        node.children[ix] = child.clone();
    }
}

impl TreeNodeChild<usize> for TTreeChild {
    fn reduce(self, mut accept: impl FnMut(&Self) -> Option<bool>) -> (Self, usize, usize, bool) {
        match self {
            Self::Scope { .. } => {
                // Nothing to do yet.
                // TODO: try to flatten nested scopes.
                (self, 0, 0, false)
            }
            Self::Choice(mut event) => {
                // Note: we minimize the choice value, not the event value.
                // Simpler choices can sometimes not lead to simpler event values
                // (e.g. unsetting the bits in the choice to make it round will not make the event value round).
                let choice = event.unwrap_choice_value();
                let (choice_min, reduced, early_exit) = reduce_num(choice, |c| {
                    event.set_choice_value(c);
                    accept(&Self::Choice(event.clone()))
                });
                event.set_choice_value(choice_min);
                (Self::Choice(event), 0, usize::from(reduced), early_exit)
            }
            Self::Repeat { .. } => {
                let (s, removed, early_exit) = reduce_seq(self, accept);
                (s, removed, 0, early_exit)
            }
        }
    }

    fn extend_vec(self, v: &mut Vec<usize>) {
        match self {
            Self::Choice { .. } => {}
            Self::Scope { id } => v.push(id),
            Self::Repeat { elements, .. } => {
                // Note: we don't push the repeat size scope.
                v.extend(elements.into_iter().flatten().map(|(id, _effect)| id));
            }
        }
    }
}

impl Seq for TTreeChild {
    fn mask(&self, begin: usize, end: usize) -> Option<(Self, usize)> {
        let Self::Repeat {
            id,
            size: Some(Event::Size { size, min, max }),
            elements,
        } = self
        else {
            unreachable!("internal error: malformed repeat");
        };
        let (mut masked, mut masked_noop) = (0, 0);
        for &(_id, effect) in elements[begin..end].iter().flatten() {
            masked += 1;
            masked_noop += usize::from(effect == Effect::Noop);
        }
        if masked == 0 {
            return None;
        }
        let masked_normal = masked - masked_noop;
        if *size - (masked_normal as u64) < *min {
            return None;
        }
        // Adjust the size to match the real number of elements
        // (it may be higher because this repeat did not finish due to panic).
        // Otherwise, we'll create tapes with size consistently larger than
        // the number of elements, which will make us use the void unnecessary.
        let size = (*size).min((elements.len() as u64).max(*min));
        // Mask both noop and normal elements, but report only the number of normal masked.
        let mut elements = elements.clone();
        elements[begin..end].fill(Option::None);
        Some((
            Self::Repeat {
                id: *id,
                size: Some(Event::Size {
                    size: size - masked_normal as u64,
                    min: *min,
                    max: *max,
                }),
                elements,
            },
            masked_normal,
        ))
    }

    fn size_min(&self) -> usize {
        let Self::Repeat {
            size: Some(Event::Size { min, .. }),
            ..
        } = self
        else {
            unreachable!("internal error: malformed repeat");
        };
        *min as usize
    }

    fn size_masked(&self) -> usize {
        let Self::Repeat { elements, .. } = self else {
            unreachable!("internal error: malformed repeat");
        };
        // Treat noop elements as pre-masked ones.
        elements
            .iter()
            .filter(|c| {
                if let Some((_, effect)) = c {
                    *effect == Effect::Noop
                } else {
                    true
                }
            })
            .count()
    }

    fn size_total(&self) -> usize {
        let Self::Repeat { elements, .. } = self else {
            unreachable!("internal error: malformed repeat");
        };
        elements.len()
    }
}
