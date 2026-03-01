#![expect(dead_code)] // TODO

use crate::label::Label;
use alloc::{vec, vec::Vec};
use core::num::NonZero;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpanId(NonZero<u32>);

impl core::fmt::Debug for SpanId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "#{}", self.index())
    }
}

impl SpanId {
    #[must_use]
    pub const fn root() -> Self {
        Self::from_index(0)
    }

    const fn from_index(value: usize) -> Self {
        assert!(value < (i32::MAX - 1) as usize);
        Self(NonZero::new(value as u32 + 1).expect("internal error: span ID overflow"))
    }

    fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

#[expect(clippy::enum_variant_names)] // TODO
enum Event {
    SpanNew { span: SpanId, parent: SpanId },
    SpanEnter { span: SpanId },
    SpanExit { span: SpanId },
    SpanLabel { span: SpanId, label: Label },
}

enum SpanKind {
    Regular,
    Repeat,
    Select,
}

struct Span {
    #[expect(clippy::struct_field_names)]
    span: SpanId,
    kind: SpanKind,
    parent: Option<SpanId>,
    depth: u16,
    index: u32,
    children_num: u32,
    child_first: Option<SpanId>, // can remove
    child_last: Option<SpanId>,
    sibling_prev: Option<SpanId>,
    sibling_next: Option<SpanId>, // can remove
    label: Label,
    // scope id?
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<Option<Span>>() <= 64);

pub struct Tracer {
    trace: Vec<Event>,
    spans: Vec<Span>, // just enough info to generate new events (scope ids etc)
}
// resolve span id (= coord) to get generation options

impl Tracer {
    #[must_use]
    #[expect(clippy::new_without_default)] // TODO
    pub fn new() -> Self {
        Self {
            trace: Vec::new(),
            spans: vec![Span {
                span: SpanId::root(),
                kind: SpanKind::Regular,
                parent: None,
                depth: 0,
                index: 0,
                children_num: 0,
                child_first: None,
                child_last: None,
                sibling_prev: None,
                sibling_next: None,
                label: Label::default(),
            }],
        }
    }

    pub fn span_new(&mut self, parent: SpanId) -> SpanId {
        debug_assert!(self.span_exists(parent));
        let span = SpanId::from_index(self.spans.len());
        let parent_depth = self.span(parent).depth;
        let parent_children = self.span(parent).children_num;
        let parent_last = self.span(parent).child_last;
        {
            let parent_span = self.span_mut(parent);
            parent_span.children_num += 1;
            if parent_span.child_first.is_none() {
                parent_span.child_first = Some(span);
            }
            parent_span.child_last = Some(span);
        }
        if let Some(prev) = parent_last {
            self.span_mut(prev).sibling_next = Some(span);
        }
        self.spans.push(Span {
            span,
            kind: SpanKind::Regular,
            parent: Some(parent),
            depth: parent_depth + 1,
            index: parent_children,
            children_num: 0,
            child_first: None,
            child_last: None,
            sibling_prev: parent_last,
            sibling_next: None,
            label: Label::default(),
        });
        self.trace.push(Event::SpanNew { span, parent });
        span
    }

    pub fn span_enter(&mut self, span: SpanId) {
        debug_assert!(self.span_exists(span));
        self.trace.push(Event::SpanEnter { span });
    }

    pub fn span_exit(&mut self, span: SpanId) {
        debug_assert!(self.span_exists(span));
        self.trace.push(Event::SpanExit { span });
    }

    pub fn span_label(&mut self, span: SpanId, label: &str) {
        debug_assert!(self.span_exists(span));
        self.span_mut(span).label = Label::from(label);
        self.trace.push(Event::SpanLabel {
            span,
            label: Label::from(label),
        });
    }

    // we only have one kind of progress per span
    pub fn span_progress_require(&mut self, span: SpanId, progress_min: u64, progress_max: u64) {
        debug_assert!(self.span_exists(span));
        debug_assert!(progress_min <= progress_max);
        // assert is not a repeat yet
        todo!()
    }

    pub fn span_progress(&mut self, span: SpanId, _progress: Option<i64>) {
        debug_assert!(self.span_exists(span));
        // assert parent is repeat
        todo!()
    }

    pub fn span_variants_num(&mut self, span: SpanId, _n: u32) {
        debug_assert!(self.span_exists(span));
        // assert is not a select yet
        todo!()
    }

    // TODO: bad, we need possibility set as a separate entity
    // TODO: use some kind of anchor?
    pub fn span_variants_label(&mut self, span: SpanId, _i: u32, _label: &str) {
        debug_assert!(self.span_exists(span));
        // assert span is select & i is in bounds
        todo!()
    }

    pub fn span_variant(&mut self, span: SpanId, _i: u32, _label: &str) {
        debug_assert!(self.span_exists(span));
        // assert parent is select & i is in bounds
        todo!()
    }
}

impl Tracer {
    fn span_exists(&self, span: SpanId) -> bool {
        span.index() < self.spans.len()
    }

    fn span(&self, span: SpanId) -> &Span {
        &self.spans[span.index()]
    }

    fn span_mut(&mut self, span: SpanId) -> &mut Span {
        &mut self.spans[span.index()]
    }
}

mod ideas {
    #![expect(unused)]
    #![expect(clippy::unused_self)]

    #[derive(Clone, Copy)]
    struct SpanId(u32);

    impl SpanId {
        const fn root() -> Self {
            Self(0)
        }
    }

    #[derive(Clone, Copy)]
    struct PropId(u32);

    #[derive(Clone, Copy)]
    enum PropKind {
        AssertUnreachable,
        AssertAny,
        AssertAll,
        CoverAny,
        CoverAll,
    }

    // TODO: track scope counters [fuzzer-like coverage] with binning; multiply by cover things
    // TODO: use scope paths to deduplicates failures? different prefix => likely different failure
    // TODO: use flow events for dependencies? start-pass-finish
    // TODO: perfetto export
    // TODO: maximize stuff!
    struct Tracer {}

    #[derive(Clone, Copy)]
    enum CompareMode {
        Less,
        LessEq,
        Eq,
        NotEq,
        GreaterEq,
        Greater,
    }

    #[derive(Clone, Copy)]
    enum StateGuidance {
        Same,
        Different,
        Small,
        Large,
    }

    #[derive(Clone, Copy)]
    enum Primitive<'a> {
        Bool(bool),
        Int(i64),
        Float(f64),
        String(&'a str),
    }

    impl Tracer {
        fn new() -> Self {
            Self {}
        }

        fn span_new(&mut self, parent: SpanId) -> SpanId {
            todo!()
        }

        fn span_enter(&mut self, span: SpanId) {}
        fn span_exit(&mut self, span: SpanId) {}
        fn span_attr_label(&mut self, span: SpanId, label: &str) {}
        fn span_attr_rejected(&mut self, span: SpanId) {}

        #[must_use]
        fn log_enabled(&self, span: SpanId) -> bool {
            todo!()
        }
        fn log_message(&mut self, span: SpanId, message: &str) {}

        // cover: cover select variants, cover repeat sizes, cover ...?
        fn prop_new(&mut self, span: SpanId, kind: PropKind) -> PropId {
            todo!()
        }
        fn prop_compare(
            &mut self,
            prop: PropId,
            label: &str,
            value: Primitive,
            target: Primitive,
            mode: CompareMode,
        ) {
        }
        #[must_use]
        fn prop_fail(&self, prop: PropId) -> bool {
            todo!()
        }
        fn prop_fail_source(
            &mut self,
            prop: PropId,
            message: &str,
            file: &str,
            line: u32,
            col: u32,
        ) {
        }

        fn state_guide(
            &mut self,
            span: SpanId,
            label: &str,
            value: Primitive,
            guidance: StateGuidance,
        ) {
        }

        // loop markers? mark span with counter; indicate possible range
        // need to special-case rejection so that next iteration span has new seed

        // select markers? mark span with variant; mark span with possible (seen) variants
        // maybe not all variants are possible! (error variant, or size overflow)

        // find markers? select-that-matches

        // cover (selected) + observe (possible)

        // grab seed from current span to seed rand-core
        // random draws, + forced
        // permutations, splits, combinations

        // input:
        // - typical (biased)
        // - unusual (swarm, corner-cases)
        // - interesting (cmplog, feedback)
    }
}
