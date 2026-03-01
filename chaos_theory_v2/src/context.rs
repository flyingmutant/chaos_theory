#![expect(clippy::missing_panics_doc)] // TODO

use core::ops::{Bound, RangeBounds};

use chaos_theory_core::{SpanId, Tracer};

pub struct Context {
    tracer: Tracer,
    current: SpanId,
}

impl Context {
    fn _new() -> Self {
        Self {
            tracer: Tracer::new(),
            current: SpanId::root(),
        }
    }

    fn enter(&mut self, label: &str) -> SpanId {
        let prev = self.current;
        let span = self.tracer.span_new(self.current);
        self.tracer.span_label(span, label);
        self.tracer.span_enter(span);
        self.current = span;
        prev
    }

    fn exit(&mut self, prev: SpanId) {
        self.tracer.span_exit(self.current);
        self.current = prev;
    }

    pub fn span<T>(&mut self, label: &str, body: impl FnOnce(&mut Self) -> T) -> T {
        let prev = self.enter(label);
        let v = body(self);
        self.exit(prev);
        v
    }

    pub fn progress_require(&mut self, progress: impl RangeBounds<u64>) {
        let min = match progress.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_sub(1).expect("invalid start bound"),
        };
        let max = match progress.end_bound() {
            Bound::Unbounded => u64::MAX,
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).expect("invalid end bound"),
        };
        assert!(min <= max, "invalid range {min:?}..={max:?}");
        self.tracer.span_progress_require(self.current, min, max);
    }

    pub fn progress(&mut self, progress: Option<i64>) {
        self.tracer.span_progress(self.current, progress);
    }

    pub fn variants(&mut self, variants: &[&str]) {
        self.tracer
            .span_variants_num(self.current, variants.len() as u32);
        for (i, v) in variants.iter().enumerate() {
            self.tracer.span_variants_label(self.current, i as u32, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "WIP"]
    fn context_smoke() {
        let mut ctx = Context::_new();

        ctx.span("a", |ctx| {
            ctx.span("b", |ctx| {
                ctx.span("c", |_ctx| {});
            });
        });

        ctx.span("loop", |ctx| {
            ctx.progress_require(2..=2); // optional
            // span here that selects the loop size?

            for _ in 0..=1 {
                // TODO: we don't really need a label for the child span (tracer does not require it)
                ctx.span("", |ctx| {
                    ctx.progress(None);
                    // TODO: break from here?
                });
            }
        });

        ctx.span("select", |ctx| {
            ctx.variants(&["left", "right"]);

            // span here that selects the variant?
            let selected_variant = "variant";

            // ????? link it up to variants
            // do we link by index or by str, or by some stable hash?
            ctx.span(selected_variant, |_ctx| match selected_variant {
                "left" | "right" => {}
                _ => unreachable!(),
            });
        });

        ctx.span("loop", |ctx| {
            ctx.progress_require(2..=2); // optional

            for _ in 0..=1 {
                ctx.span("iter", |ctx| {
                    ctx.progress(None);
                    // iter == select, mark variants here
                    // TODO: child span
                });
            }
        });
    }
}
