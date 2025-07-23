// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::sync::Arc;
use core::{fmt::Debug, fmt::Display, num::NonZero};

use crate::{Arbitrary, Effect, Generator, SourceRaw, make, tape::TapeMeta, varint};

const BUFFER_TOO_SHORT: &str = "buffer too short";
const INVALID_UTF_8: &str = "invalid UTF-8 data";
const INVALID_U32: &str = "invalid u32 value";
const INVALID_BOOL: &str = "invalid boolean value";

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ScopeKind {
    Plain,
    RepeatSize,
    RepeatElement,
    SelectIndex,
    SelectVariant,
}

impl Debug for ScopeKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Plain => f.write_str("P"),
            Self::RepeatSize => f.write_str("R#"),
            Self::RepeatElement => f.write_str("RE"),
            Self::SelectIndex => f.write_str("S#"),
            Self::SelectVariant => f.write_str("SV"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct InternId(pub(crate) u32); // 0 => empty string; i + 1 => string at index i

impl Display for InternId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "@{}", self.0)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum Event {
    ScopeStart {
        id: u64,
        kind: ScopeKind,
        effect: Effect,
        meta: Option<ScopeStartMeta>,
    },
    ScopeEnd,
    Size {
        size: u64,
        min: u64,
        max: u64,
    },
    Index {
        index: u64,
        max: u64,
        forced: bool,
    },
    Value {
        value: u64,
        min: u64,
        max: u64,
    },
    Meta(MetaEvent),
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ScopeStartMeta {
    pub label: InternId,
    pub variant: InternId,
    pub variant_semantic: bool,
    pub variant_index: u32,
    pub counter: u32,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum MetaEvent {
    Intern {
        id: InternId, // redundant, but costs nothing and is useful for debug
        value: Arc<str>,
    },
    // TODO(meta): cover
    // TODO(meta): select variants
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<Option<Event>>() == 32);

impl Event {
    pub(crate) fn unwrap_choice_value(&self) -> u64 {
        self.choice_value()
            .expect("internal error: getting choice for non-choice event")
    }

    pub(crate) fn choice_value(&self) -> Option<u64> {
        match self {
            Self::Size { size, min, .. } => Some(*size - *min),
            Self::Value { value, min, .. } => Some(*value - *min),
            Self::Index { index, .. } => Some(*index),
            _ => None,
        }
    }

    pub(crate) fn set_choice_value(&mut self, choice: u64) {
        let v = match self {
            Self::Size { min, .. } | Self::Value { min, .. } => choice + *min,
            Self::Index { .. } => choice,
            _ => unreachable!("internal error: setting choice for non-choice event"),
        };
        self.set_value(v);
    }

    pub(crate) fn set_value(&mut self, v: u64) {
        match self {
            Self::Size { size, min, max } => {
                debug_assert!((*min..=*max).contains(&v));
                *size = v;
            }
            Self::Value { value, min, max } => {
                debug_assert!((*min..=*max).contains(&v));
                *value = v;
            }
            Self::Index { index, max, .. } => {
                debug_assert!(v <= *max);
                *index = v;
            }
            _ => unreachable!("internal error: setting value for non-choice event"),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::ScopeStart {
                id,
                kind,
                effect,
                meta,
            } => {
                if let Some(meta) = meta
                    && meta.counter != 0
                    && *kind != ScopeKind::RepeatElement
                {
                    return Err("non-zero counter not on a repeat element");
                }
                if *id == 0 {
                    Err("zero scope ID")
                } else if *effect != Effect::Success && *kind != ScopeKind::RepeatElement {
                    Err("non-success effect not on a repeat element")
                } else {
                    Ok(())
                }
            }
            Self::ScopeEnd | Self::Meta(MetaEvent::Intern { .. }) => Ok(()),
            Self::Size { size, min, max } => {
                if min > max {
                    Err("size min > size max")
                } else if size < min || size > max {
                    Err("size not between min and max")
                } else {
                    Ok(())
                }
            }
            Self::Index { index, max, .. } => {
                if index > max {
                    Err("index out of bounds")
                } else {
                    Ok(())
                }
            }
            Self::Value { value, min, max } => {
                if min > max {
                    Err("value min > value max")
                } else if value < min || value > max {
                    Err("value not between min and max")
                } else {
                    Ok(())
                }
            }
        }
    }

    pub(crate) fn max_size(&self) -> usize {
        // Try to be reasonably quick and simple.
        if let Self::Meta(MetaEvent::Intern { id: _id, value }) = self {
            1 + varint::MAX_SIZE * 2 + value.len()
        } else {
            1 + varint::MAX_SIZE * 3
        }
    }

    const ENCODE_SCOPE_START_PLAIN: u8 = b'P';
    const ENCODE_SCOPE_START_REPEAT_SIZE: u8 = b'S';
    const ENCODE_SCOPE_START_REPEAT_ELEMENT: u8 = b'E';
    const ENCODE_SCOPE_START_REPEAT_ELEMENT_NOOP: u8 = b'N';
    const ENCODE_SCOPE_START_REPEAT_ELEMENT_CHANGE: u8 = b'C';
    const ENCODE_SCOPE_START_SELECT_INDEX: u8 = b'I';
    const ENCODE_SCOPE_START_SELECT_VARIANT: u8 = b'V';
    const ENCODE_SCOPE_END: u8 = b'Y';
    const ENCODE_SIZE: u8 = b's';
    const ENCODE_SIZE_UNBOUNDED: u8 = b't';
    const ENCODE_INDEX: u8 = b'i';
    const ENCODE_INDEX_FORCED: u8 = b'f';
    const ENCODE_VALUE: u8 = b'v';
    const ENCODE_VALUE_UNBOUNDED: u8 = b'u';
    const ENCODE_META_INTERN: u8 = b'Z';

    pub(crate) fn save<'buf>(
        &self,
        mut buf: &'buf mut [u8],
        ignore_meta: bool,
    ) -> Result<&'buf mut [u8], &'static str> {
        match self {
            Self::ScopeStart {
                id,
                kind,
                effect,
                meta,
            } => {
                let id_size = size_of_val(id);
                if buf.len() < 1 + id_size {
                    return Err(BUFFER_TOO_SHORT);
                }
                let marker = match *kind {
                    ScopeKind::Plain => Self::ENCODE_SCOPE_START_PLAIN,
                    ScopeKind::RepeatSize => Self::ENCODE_SCOPE_START_REPEAT_SIZE,
                    ScopeKind::RepeatElement => match *effect {
                        Effect::Noop => Self::ENCODE_SCOPE_START_REPEAT_ELEMENT_NOOP,
                        Effect::Change => Self::ENCODE_SCOPE_START_REPEAT_ELEMENT_CHANGE,
                        Effect::Success => Self::ENCODE_SCOPE_START_REPEAT_ELEMENT,
                    },
                    ScopeKind::SelectIndex => Self::ENCODE_SCOPE_START_SELECT_INDEX,
                    ScopeKind::SelectVariant => Self::ENCODE_SCOPE_START_SELECT_VARIANT,
                };
                buf[0] = marker;
                buf = &mut buf[1..];
                buf[..id_size].copy_from_slice(&id.to_le_bytes());
                buf = &mut buf[id_size..];
                if let Some(meta) = meta
                    && !ignore_meta
                {
                    buf = varint::encode(1, buf)?; // TODO(meta): write length here instead to have self-synchronizing format?
                    buf = Self::save_scope_start_meta(meta, buf)?;
                } else {
                    buf = varint::encode(0, buf)?;
                }
                Ok(buf)
            }
            Self::ScopeEnd => {
                if buf.is_empty() {
                    return Err(BUFFER_TOO_SHORT);
                }
                buf[0] = Self::ENCODE_SCOPE_END;
                Ok(&mut buf[1..])
            }
            Self::Size { size, min, max } => Self::save_choice_triple(
                Self::ENCODE_SIZE,
                Self::ENCODE_SIZE_UNBOUNDED,
                *size,
                *min,
                *max,
                buf,
            ),
            Self::Index { index, max, forced } => Self::save_choice_triple(
                Self::ENCODE_INDEX,
                Self::ENCODE_INDEX_FORCED,
                *index,
                *max,
                if *forced { u64::MAX } else { 0 }, // yep, a hack
                buf,
            ),
            Self::Value { value, min, max } => Self::save_choice_triple(
                Self::ENCODE_VALUE,
                Self::ENCODE_VALUE_UNBOUNDED,
                *value,
                *min,
                *max,
                buf,
            ),
            Self::Meta(meta) => {
                if ignore_meta {
                    return Ok(buf);
                }
                if buf.is_empty() {
                    return Err(BUFFER_TOO_SHORT);
                }
                match meta {
                    MetaEvent::Intern { id, value } => {
                        buf[0] = Self::ENCODE_META_INTERN;
                        buf = &mut buf[1..];
                        buf = varint::encode(u64::from(id.0), buf)?;
                        let n = value.len();
                        buf = varint::encode(n as u64, buf)?;
                        if buf.len() < n {
                            return Err(BUFFER_TOO_SHORT);
                        }
                        buf[..n].copy_from_slice(value.as_bytes());
                        Ok(&mut buf[n..])
                    }
                }
            }
        }
    }

    fn save_scope_start_meta<'buf>(
        meta: &ScopeStartMeta,
        mut buf: &'buf mut [u8],
    ) -> Result<&'buf mut [u8], &'static str> {
        buf = varint::encode(u64::from(meta.label.0), buf)?;
        buf = varint::encode(u64::from(meta.variant.0), buf)?;
        buf = varint::encode(u64::from(meta.variant_semantic), buf)?;
        buf = varint::encode(u64::from(meta.variant_index), buf)?;
        buf = varint::encode(u64::from(meta.counter), buf)?;
        Ok(buf)
    }

    fn save_choice_triple(
        marker: u8,
        marker_unbounded: u8,
        value: u64,
        min: u64,
        max: u64,
        mut buf: &mut [u8],
    ) -> Result<&mut [u8], &'static str> {
        if buf.is_empty() {
            return Err(BUFFER_TOO_SHORT);
        }
        let unbounded = max == u64::MAX;
        buf[0] = if unbounded { marker_unbounded } else { marker };
        buf = &mut buf[1..];
        buf = varint::encode(value, buf)?;
        buf = varint::encode(min, buf)?;
        if !unbounded {
            buf = varint::encode(max, buf)?;
        }
        Ok(buf)
    }

    pub(crate) fn load(buf: &[u8], validate: bool) -> Result<(Self, &[u8]), &'static str> {
        if buf.is_empty() {
            return Err(BUFFER_TOO_SHORT);
        }
        let marker = buf[0];
        let buf = &buf[1..];
        let (event, buf) = match marker {
            Self::ENCODE_SCOPE_START_PLAIN => {
                Self::load_scope_start(ScopeKind::Plain, Effect::Success, buf)
            }
            Self::ENCODE_SCOPE_START_REPEAT_SIZE => {
                Self::load_scope_start(ScopeKind::RepeatSize, Effect::Success, buf)
            }
            Self::ENCODE_SCOPE_START_REPEAT_ELEMENT_NOOP => {
                Self::load_scope_start(ScopeKind::RepeatElement, Effect::Noop, buf)
            }
            Self::ENCODE_SCOPE_START_REPEAT_ELEMENT_CHANGE => {
                Self::load_scope_start(ScopeKind::RepeatElement, Effect::Change, buf)
            }
            Self::ENCODE_SCOPE_START_REPEAT_ELEMENT => {
                Self::load_scope_start(ScopeKind::RepeatElement, Effect::Success, buf)
            }
            Self::ENCODE_SCOPE_START_SELECT_INDEX => {
                Self::load_scope_start(ScopeKind::SelectIndex, Effect::Success, buf)
            }
            Self::ENCODE_SCOPE_START_SELECT_VARIANT => {
                Self::load_scope_start(ScopeKind::SelectVariant, Effect::Success, buf)
            }
            Self::ENCODE_SCOPE_END => Ok((Self::ScopeEnd, buf)),
            Self::ENCODE_SIZE | Self::ENCODE_SIZE_UNBOUNDED => {
                let (size, min, max, buf) =
                    Self::load_choice_triple(marker == Self::ENCODE_SIZE_UNBOUNDED, buf)?;
                Ok((Self::Size { size, min, max }, buf))
            }
            Self::ENCODE_INDEX | Self::ENCODE_INDEX_FORCED => {
                let forced = marker == Self::ENCODE_INDEX_FORCED;
                let (index, max, _, buf) = Self::load_choice_triple(forced, buf)?;
                Ok((Self::Index { index, max, forced }, buf))
            }
            Self::ENCODE_VALUE | Self::ENCODE_VALUE_UNBOUNDED => {
                let (value, min, max, buf) =
                    Self::load_choice_triple(marker == Self::ENCODE_VALUE_UNBOUNDED, buf)?;
                Ok((Self::Value { value, min, max }, buf))
            }
            Self::ENCODE_META_INTERN => {
                let (id, buf) = Self::load_u32(buf)?;
                let (n, buf) = varint::decode(buf)?;
                let n = n as usize;
                if buf.len() < n {
                    return Err(BUFFER_TOO_SHORT);
                }
                let value = Vec::from(&buf[..n]);
                let Ok(value) = String::from_utf8(value) else {
                    return Err(INVALID_UTF_8);
                };
                Ok((
                    Self::Meta(MetaEvent::Intern {
                        id: InternId(id),
                        value: value.into(),
                    }),
                    &buf[n..],
                ))
            }
            _ => Err("unexpected event marker"),
        }?;
        if validate {
            event.validate()?;
        }
        Ok((event, buf))
    }

    fn load_bool(buf: &[u8]) -> Result<(bool, &[u8]), &'static str> {
        let (b, buf) = varint::decode(buf)?;
        match b {
            0 => Ok((false, buf)),
            1 => Ok((true, buf)),
            _ => Err(INVALID_BOOL),
        }
    }

    fn load_u32(buf: &[u8]) -> Result<(u32, &[u8]), &'static str> {
        let (id, buf) = varint::decode(buf)?;
        let Ok(id) = u32::try_from(id) else {
            return Err(INVALID_U32);
        };
        Ok((id, buf))
    }

    fn load_scope_start(
        kind: ScopeKind,
        effect: Effect,
        buf: &[u8],
    ) -> Result<(Self, &[u8]), &'static str> {
        const ID_SIZE: usize = size_of::<u64>();
        if buf.len() < ID_SIZE {
            return Err(BUFFER_TOO_SHORT);
        }
        let id = u64::from_le_bytes(
            buf[..ID_SIZE]
                .try_into()
                .expect("internal error: mismatched slice/array size"),
        );
        let buf = &buf[ID_SIZE..];
        let (has_meta, mut buf) = Self::load_bool(buf)?;
        let mut meta_opt = None;
        if has_meta {
            let meta;
            (meta, buf) = Self::load_scope_start_meta(buf)?;
            meta_opt = Some(meta);
        }
        Ok((
            Self::ScopeStart {
                id,
                kind,
                effect,
                meta: meta_opt,
            },
            buf,
        ))
    }

    fn load_scope_start_meta(buf: &[u8]) -> Result<(ScopeStartMeta, &[u8]), &'static str> {
        let (label, buf) = Self::load_u32(buf)?;
        let (variant, buf) = Self::load_u32(buf)?;
        let (variant_semantic, buf) = Self::load_bool(buf)?;
        let (variant_index, buf) = Self::load_u32(buf)?;
        let (counter, buf) = Self::load_u32(buf)?;
        Ok((
            ScopeStartMeta {
                label: InternId(label),
                variant: InternId(variant),
                variant_semantic,
                variant_index,
                counter,
            },
            buf,
        ))
    }

    fn load_choice_triple(
        unbounded: bool,
        buf: &[u8],
    ) -> Result<(u64, u64, u64, &[u8]), &'static str> {
        let (value, buf) = varint::decode(buf)?;
        let (min, buf) = varint::decode(buf)?;
        if unbounded {
            Ok((value, min, u64::MAX, buf))
        } else {
            let (max, buf) = varint::decode(buf)?;
            Ok((value, min, max, buf))
        }
    }

    pub(crate) fn format(
        &self,
        f: &mut core::fmt::Formatter<'_>,
        tape_meta: Option<&TapeMeta>,
    ) -> core::fmt::Result {
        match self {
            Self::ScopeStart {
                id,
                kind,
                effect,
                meta,
            } => {
                let open = match effect {
                    Effect::Noop => "<!",
                    Effect::Change => "<?",
                    Effect::Success => "<",
                };
                if f.alternate() {
                    write!(f, "{open}{kind:?}/{id:x}")?;
                } else {
                    let id_start = id >> 56;
                    let id_end = id & 0xff;
                    write!(f, "{open}{kind:?}/{id_start:02x}..{id_end:02x}")?;
                }
                if let Some(meta) = meta {
                    write!(f, " ")?;
                    meta.format(f, tape_meta)?;
                }
                Ok(())
            }
            Self::ScopeEnd => f.write_str(">"),
            Self::Size { size, min, max } => {
                if *max == u64::MAX {
                    write!(f, "S({size}, {min}..)")
                } else {
                    write!(f, "S({size}, {min}..={max})")
                }
            }
            Self::Index { index, max, forced } => {
                if *forced {
                    write!(f, "I!({index}, ..={max})")
                } else {
                    write!(f, "I({index}, ..={max})")
                }
            }
            Self::Value { value, min, max } => {
                if *max == u64::MAX {
                    write!(f, "V({value}, {min}..)")
                } else {
                    write!(f, "V({value}, {min}..={max})")
                }
            }
            Self::Meta(MetaEvent::Intern { id, value }) => {
                write!(f, "Z({id} -> {value:?})")
            }
        }
    }
}

impl Debug for Event {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.format(f, None)
    }
}

impl Arbitrary for Event {
    fn arbitrary() -> impl Generator<Item = Self> {
        make::from_fn(make_event)
    }
}

// TODO(meta): use (fill during generation) an optional mutable TapeMeta to generate valid-in-context events
#[expect(clippy::too_many_lines)]
fn make_event(src: &mut SourceRaw, example: Option<&Event>) -> Event {
    let example_index = example.map(|e| match e {
        Event::ScopeStart { .. } => 0,
        Event::ScopeEnd => 1,
        Event::Size { .. } => 2,
        Event::Index { .. } => 3,
        Event::Value { .. } => 4,
        Event::Meta(MetaEvent::Intern { .. }) => 5,
    });
    let variants = &["start", "end", "size", "index", "value", "meta_intern"];
    let variants_num = NonZero::new(variants.len()).expect("internal error: no variants");
    src.select(
        "<event>",
        example_index,
        variants_num,
        |ix| variants[ix],
        |src, v, _ix| match v {
            "start" => {
                let example = match example {
                    Some(Event::ScopeStart {
                        id,
                        kind,
                        effect,
                        meta,
                    }) => Some((id, kind, effect, meta)),
                    _ => None,
                };
                let example_kind_index = example.map(|e| match *e.1 {
                    ScopeKind::Plain => 0,
                    ScopeKind::RepeatSize => 1,
                    ScopeKind::RepeatElement => 2,
                    ScopeKind::SelectIndex => 3,
                    ScopeKind::SelectVariant => 4,
                });
                let id = src.any_of(
                    "id",
                    make::arbitrary().filter_assume(|id| *id != 0),
                    example.map(|e| e.0),
                );
                let (&kind, _) = src
                    .choose(
                        "kind",
                        example_kind_index,
                        &[
                            ScopeKind::Plain,
                            ScopeKind::RepeatSize,
                            ScopeKind::RepeatElement,
                            ScopeKind::SelectIndex,
                            ScopeKind::SelectVariant,
                        ],
                    )
                    .expect("internal error: no scope kinds");
                let example_effect = if kind == ScopeKind::RepeatElement {
                    example.map(|e| e.2)
                } else {
                    Some(&Effect::Success)
                };
                let effect = src.any("effect", example_effect);
                let meta = src.maybe("meta", example.map(|e| e.3.is_some()), |src| {
                    // Inline, since meta generation depends on kind, and Arbitrary implementation don't have access to context.
                    let example = example.and_then(|e| e.3.as_ref());
                    let label = InternId(src.any("label", example.map(|e| &e.label.0)));
                    let variant = InternId(src.any("variant", example.map(|e| &e.variant.0)));
                    let variant_semantic =
                        src.any("variant_semantic", example.map(|e| &e.variant_semantic));
                    let variant_index = src.any("variant_index", example.map(|e| &e.variant_index));
                    let example_counter = if kind == ScopeKind::RepeatElement {
                        example.map(|e| &e.counter)
                    } else {
                        Some(&0)
                    };
                    let counter = src.any("counter", example_counter);
                    ScopeStartMeta {
                        label,
                        variant,
                        variant_semantic,
                        variant_index,
                        counter,
                    }
                });
                Event::ScopeStart {
                    id,
                    kind,
                    effect,
                    meta,
                }
            }
            "end" => Event::ScopeEnd,
            "size" => {
                let example = match example {
                    Some(Event::Size { size, min, max }) => Some((size, min, max)),
                    _ => None,
                };
                let min = src.any("min", example.map(|e| e.1));
                let max = src.any_of("max", make::int_in_range(min..), example.map(|e| e.2));
                let size = src.any_of("size", make::int_in_range(min..=max), example.map(|e| e.0));
                Event::Size { size, min, max }
            }
            "index" => {
                let example = match example {
                    Some(Event::Index { index, max, forced }) => Some((index, max, forced)),
                    _ => None,
                };
                let max = src.any_of("max", make::int_in_range(..), example.map(|e| e.1));
                let index = src.any_of("index", make::int_in_range(..=max), example.map(|e| e.0));
                let forced = src.any("forced", example.map(|e| e.2));
                Event::Index { index, max, forced }
            }
            "value" => {
                let example = match example {
                    Some(Event::Value { value, min, max }) => Some((value, min, max)),
                    _ => None,
                };
                let min = src.any("min", example.map(|e| e.1));
                let max = src.any_of("max", make::int_in_range(min..), example.map(|e| e.2));
                let value =
                    src.any_of("value", make::int_in_range(min..=max), example.map(|e| e.0));
                Event::Value { value, min, max }
            }
            "meta_intern" => {
                let example = match example {
                    Some(Event::Meta(MetaEvent::Intern { id, value })) => Some((id, value)),
                    _ => None,
                };
                let id = InternId(src.any("id", example.map(|e| &e.0.0)));
                let value = src.any("value", example.map(|e| e.1));
                Event::Meta(MetaEvent::Intern { id, value })
            }
            _ => unreachable!(),
        },
    )
}

impl ScopeStartMeta {
    fn format(
        &self,
        f: &mut core::fmt::Formatter<'_>,
        tape_meta: Option<&TapeMeta>,
    ) -> core::fmt::Result {
        write!(f, "[[")?;
        if let Some(tape_meta) = tape_meta
            && let Some(label) = tape_meta.get(self.label)
        {
            write!(f, "{label}")?;
        }
        write!(f, "{}", self.label)?;
        if self.variant_semantic {
            write!(f, " ")?;
        } else {
            write!(f, " ~")?;
        }
        if self.variant.0 == 0 {
            write!(f, "{}", self.variant_index)?;
        } else {
            if let Some(tape_meta) = tape_meta
                && let Some(variant) = tape_meta.get(self.variant)
            {
                write!(f, "{variant}")?;
            }
            write!(f, "{}", self.variant)?;
        }
        write!(f, " #{}]]", self.counter)
    }
}

impl Debug for ScopeStartMeta {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.format(f, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, make, tests::prop_smoke, vdbg};

    #[test]
    fn event_arbitrary_valid() {
        check(|src| {
            let event: Event = src.any("event");
            event.validate().unwrap();
        });
    }

    #[test]
    fn event_smoke() {
        check(|src| {
            prop_smoke(src, "Event", make::arbitrary::<Event>());
        });
    }

    #[test]
    fn event_save_load_roundtrip() {
        check(|src| {
            let event: Event = src.any("event");
            vdbg!(src, &event);
            let mut buf = vec![0; event.max_size()];
            let rem_len = event.save(&mut buf, false).unwrap().len();
            vdbg!(src, &buf);
            let (event_, _) = Event::load(&buf[..buf.len() - rem_len], true).unwrap();
            assert_eq!(event, event_);
        });
    }
}
