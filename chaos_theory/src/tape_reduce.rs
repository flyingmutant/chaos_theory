// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::cmp::Ordering;

use crate::{
    Effect,
    reduce::{Seq, Tree, TreeNodeChild, reduce_num, reduce_seq, visit_seq_candidates},
    tape::Tape,
    tape_event::{Event, ScopeKind},
};

type TRepeatElement = Option<(usize, Effect, bool)>;

#[derive(Debug)]
struct TTreeNode {
    scope_id: u64,
    scope_kind: ScopeKind,
    scope_effect: Effect,
    scope_discardable: bool,
    parent_id: Option<usize>,
    children: Vec<TTreeChild>,
}

impl TTreeNode {
    fn new(
        parent_id: Option<usize>,
        scope_id: u64,
        scope_kind: ScopeKind,
        scope_effect: Effect,
        scope_discardable: bool,
    ) -> Self {
        Self {
            scope_id,
            scope_kind,
            scope_effect,
            scope_discardable,
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
        elements: Vec<TRepeatElement>,
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
                    elements.push(Some((id, node.scope_effect, node.scope_discardable)));
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
        let root = TTreeNode::new(None, 0, ScopeKind::Plain, Effect::Success, false);
        let mut cur_node_id = Some(self.add_node(root));
        debug_assert_eq!(cur_node_id, Some(0));
        let mut fixup_repeat_size = false;
        for event in events {
            match event {
                Event::ScopeStart {
                    id,
                    kind,
                    effect,
                    discardable,
                    meta: _,
                } => {
                    let node = TTreeNode::new(cur_node_id, *id, *kind, *effect, *discardable);
                    cur_node_id = Some(self.add_node(node));
                    fixup_repeat_size = *kind == ScopeKind::RepeatSize;
                }
                Event::ScopeEnd => {
                    let node = self.node_mut(cur_node_id);
                    debug_assert!(node.parent_id.is_some());
                    cur_node_id = node.parent_id;
                }
                Event::Size { .. }
                | Event::Index { .. }
                | Event::Value { .. }
                | Event::Token { .. }
                | Event::Observe { .. } => {
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
                discardable: node.scope_discardable,
                meta: None,
            });
        }
        for child in &node.children {
            match child {
                TTreeChild::Choice(event) => events.push(event.clone()),
                TTreeChild::Scope { id } => self.to_tape_rec(events, *id, ignore_noop),
                TTreeChild::Repeat { id, elements, .. } => {
                    self.to_tape_rec(events, *id, ignore_noop);
                    for (id, effect, discardable) in elements.iter().flatten() {
                        if !(ignore_noop && *effect == Effect::Noop && *discardable) {
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

    // TODO: make this work not only on repeat elements (e.g. tuple/struct fields)
    pub(crate) fn sort_child(
        &mut self,
        node_id: usize,
        ix: usize,
        choice_indices: &mut Vec<usize>,
        accept: &mut impl FnMut(&Self) -> Option<bool>,
    ) -> bool {
        let TTreeChild::Repeat { elements, .. } = &self.nodes[node_id].children[ix] else {
            return false;
        };
        if elements.len() < 2 {
            return false;
        }
        let elements = elements.clone();
        // Reuse the dense node lookup across repeats, while keeping their choice data local.
        if choice_indices.len() != self.nodes.len() {
            choice_indices.resize(self.nodes.len(), 0);
        }
        let mut choices = Vec::with_capacity(elements.len());
        for (id, _effect, _discardable) in elements.iter().flatten() {
            let mut element_choices = Vec::new();
            self.collect_node_choices(*id, &mut element_choices);
            choice_indices[*id] = choices.len();
            choices.push(element_choices);
        }

        let mut early_exit = false;
        visit_seq_candidates(elements.len(), false, |begin, end| {
            let original = self.repeat_elements(node_id, ix)[begin..end].to_vec();
            let mut sorted = original.clone();
            Self::sort_repeat_elements(&mut sorted, choice_indices, &choices);
            if sorted == original {
                return true;
            }
            early_exit = self.try_reorder(node_id, ix, begin, &original, &sorted, accept);
            !early_exit
        });
        if early_exit {
            return true;
        }
        false
    }

    fn repeat_elements(&self, node_id: usize, ix: usize) -> &[TRepeatElement] {
        let TTreeChild::Repeat { elements, .. } = &self.nodes[node_id].children[ix] else {
            unreachable!("internal error: sorting non-repeat child");
        };
        elements
    }

    fn repeat_elements_mut(&mut self, node_id: usize, ix: usize) -> &mut [TRepeatElement] {
        let TTreeChild::Repeat { elements, .. } = &mut self.nodes[node_id].children[ix] else {
            unreachable!("internal error: sorting non-repeat child");
        };
        elements
    }

    fn collect_node_choices(&self, id: usize, choices: &mut Vec<u64>) {
        for child in &self.nodes[id].children {
            match child {
                TTreeChild::Choice(event) => choices.extend(event.choice_value()),
                TTreeChild::Scope { id } => self.collect_node_choices(*id, choices),
                TTreeChild::Repeat { id, elements, .. } => {
                    self.collect_node_choices(*id, choices);
                    for (id, _effect, _discardable) in elements.iter().flatten() {
                        self.collect_node_choices(*id, choices);
                    }
                }
            }
        }
    }

    fn repeat_element_choices<'a>(
        element: &TRepeatElement,
        choice_indices: &[usize],
        choices: &'a [Vec<u64>],
    ) -> &'a [u64] {
        let Some((id, _effect, _discardable)) = element else {
            return &[];
        };
        &choices[choice_indices[*id]]
    }

    fn repeat_element_cmp(
        a: &TRepeatElement,
        b: &TRepeatElement,
        choice_indices: &[usize],
        choices: &[Vec<u64>],
    ) -> Ordering {
        let a = Self::repeat_element_choices(a, choice_indices, choices);
        let b = Self::repeat_element_choices(b, choice_indices, choices);
        // Order elements by whichever concatenation places the smaller one first.
        a.iter().chain(b).cmp(b.iter().chain(a))
    }

    fn sort_repeat_elements(
        elements: &mut [TRepeatElement],
        choice_indices: &[usize],
        choices: &[Vec<u64>],
    ) {
        // Empty elements compare equal to everything, so keep their slots fixed.
        let mut nonempty: Vec<_> = elements
            .iter()
            .copied()
            .filter(|element| {
                !Self::repeat_element_choices(element, choice_indices, choices).is_empty()
            })
            .collect();
        nonempty.sort_by(|a, b| Self::repeat_element_cmp(a, b, choice_indices, choices));
        let mut nonempty = nonempty.into_iter();
        for element in elements {
            if !Self::repeat_element_choices(element, choice_indices, choices).is_empty() {
                *element = nonempty
                    .next()
                    .expect("internal error: missing sorted repeat element");
            }
        }
    }

    fn try_reorder(
        &mut self,
        node_id: usize,
        ix: usize,
        begin: usize,
        original: &[TRepeatElement],
        reordered: &[TRepeatElement],
        accept: &mut impl FnMut(&Self) -> Option<bool>,
    ) -> bool {
        let end = begin + original.len();
        self.repeat_elements_mut(node_id, ix)[begin..end].copy_from_slice(reordered);
        let accepted = accept(self);
        if accepted != Some(true) {
            self.repeat_elements_mut(node_id, ix)[begin..end].copy_from_slice(original);
        }
        accepted.is_none()
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
                if matches!(
                    event,
                    Event::Index { forced: true, .. } | Event::Token { .. } | Event::Observe { .. }
                ) {
                    return (Self::Choice(event), 0, 0, false);
                }
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
                v.extend(
                    elements
                        .into_iter()
                        .flatten()
                        .filter(|(_id, effect, discardable)| {
                            !(*effect == Effect::Noop && *discardable)
                        })
                        .map(|(id, _effect, _discardable)| id),
                );
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
        let (mut masked, mut masked_discardable_noop, mut masked_success) = (0, 0, 0);
        for &(_id, effect, discardable) in elements[begin..end].iter().flatten() {
            masked += 1;
            masked_discardable_noop += usize::from(effect == Effect::Noop && discardable);
            masked_success += usize::from(effect == Effect::Success);
        }
        if masked == 0 {
            return None;
        }
        let masked_remaining = masked - masked_discardable_noop;
        // Repeat size tracks successful elements only.
        if (masked_success as u64) > *size || *size - (masked_success as u64) < *min {
            return None;
        }
        // Adjust the size to match the real number of elements
        // (it may be higher because this repeat did not finish due to panic).
        // Otherwise, we'll create tapes with size consistently larger than
        // the number of elements, which will make us use the void unnecessary.
        let size = (*size).min((elements.len() as u64).max(*min));
        // Mask elements in the range, but only count ones that weren't pre-masked (discardable noop).
        let mut elements = elements.clone();
        elements[begin..end].fill(Option::None);
        Some((
            Self::Repeat {
                id: *id,
                size: Some(Event::Size {
                    size: size - masked_success as u64,
                    min: *min,
                    max: *max,
                }),
                elements,
            },
            masked_remaining,
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
        // Treat discardable noop elements as pre-masked ones.
        elements
            .iter()
            .filter(|c| {
                if let Some((_, effect, discardable)) = c {
                    *effect == Effect::Noop && *discardable
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, make};

    fn scope(
        id: u64,
        kind: ScopeKind,
        choices: impl IntoIterator<Item = Event>,
    ) -> impl Iterator<Item = Event> {
        core::iter::once(Event::ScopeStart {
            id,
            kind,
            effect: Effect::Success,
            discardable: false,
            meta: None,
        })
        .chain(choices)
        .chain(core::iter::once(Event::ScopeEnd))
    }

    fn repeat_tree<T: Copy + Into<u64>>(elements: &[Vec<T>]) -> TTree {
        let mut events = Vec::new();
        events.extend(scope(
            1,
            ScopeKind::RepeatSize,
            [Event::Size {
                size: elements.len() as u64,
                min: 0,
                max: elements.len() as u64,
            }],
        ));
        for (ix, choices) in elements.iter().enumerate() {
            events.extend(scope(
                ix as u64 + 2,
                ScopeKind::RepeatElement,
                choices.iter().copied().map(|value| Event::Value {
                    value: value.into(),
                    min: 0,
                    max: u64::MAX,
                }),
            ));
        }
        TTree::from_events(&events)
    }

    #[test]
    fn sort_repeat_elements() {
        check(|src| {
            let elements = src.any_of(
                "elements",
                make::vec_with_size(make::vec_with_size(make::arbitrary::<u8>(), ..5), 2..),
            );
            let mut expected: Vec<_> = (0..elements.len()).collect();
            let mut nonempty: Vec<_> = expected
                .iter()
                .copied()
                .filter(|ix| !elements[*ix].is_empty())
                .collect();
            nonempty.sort_by(|a, b| {
                elements[*a]
                    .iter()
                    .chain(&elements[*b])
                    .cmp(elements[*b].iter().chain(&elements[*a]))
            });
            let mut nonempty = nonempty.into_iter();
            for ix in &mut expected {
                if !elements[*ix].is_empty() {
                    *ix = nonempty.next().unwrap();
                }
            }
            let mut tree = repeat_tree(&elements);
            let mut choice_indices = Vec::new();
            assert!(!tree.sort_child(0, 0, &mut choice_indices, &mut |_| Some(true)));
            let actual: Vec<_> = tree
                .repeat_elements(0, 0)
                .iter()
                .flatten()
                .map(|(id, _effect, _discardable)| id - 2)
                .collect();
            assert_eq!(actual, expected);

            let base = u64::from(src.any::<u32>("base"));
            let original = [base + 4, base + 3, base + 2, base + 1];
            let chunk = [base + 4, base + 3, base + 1, base + 2];
            let adjacent = [base + 4, base + 1, base + 3, base + 2];
            let mut tree = repeat_tree(&original.map(|value| vec![value]));
            let mut choice_indices = Vec::new();
            let mut accept = |t: &TTree| {
                let tape = t.to_tape(false);
                let choices = &tape.as_choices()[1..];
                Some(
                    [original, chunk, adjacent]
                        .iter()
                        .any(|order| choices == order),
                )
            };
            assert!(!tree.sort_child(0, 0, &mut choice_indices, &mut accept));
            assert_eq!(&tree.to_tape(false).as_choices()[1..], adjacent);
        });
    }
}
