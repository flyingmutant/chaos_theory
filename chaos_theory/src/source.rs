// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#[cfg(test)]
use core::time::Duration;
use core::{fmt::Debug, num::NonZero, ops::RangeBounds};

#[cfg(test)]
use crate::tape::Tape;
#[expect(clippy::unused_trait_names, reason = "docstring references")]
use crate::{
    __panic_assume, Arbitrary, Effect, Env, Generator, MAX_SIZE, OptionExt, Scope, Tweak,
    range::SizeRange, tape_event::ScopeKind,
};

const REPEAT_REJECT_CONSEQ_SOFT_MAX: u32 = 8;
const REPEAT_REJECT_CONSEQ_HARD_MAX: u32 = MAX_SIZE as u32 + 1; // TODO: add a per-check global limit as well

const UNABLE_PERFORM_MINIMUM_REPEAT: &str =
    "unable to perform minimum number of repeat steps successfully";

/// Primary interface for working with pseudo-random data.
#[derive(Debug)]
pub struct Source<'env> {
    raw: SourceRaw<'env>,
}

impl AsRef<Env> for Source<'_> {
    fn as_ref(&self) -> &Env {
        self.raw.env
    }
}

impl AsMut<Env> for Source<'_> {
    fn as_mut(&mut self) -> &mut Env {
        self.raw.env
    }
}

impl<'env> Source<'env> {
    pub(crate) fn new(env: &'env mut Env) -> Self {
        Self {
            raw: SourceRaw { env },
        }
    }

    /// Create an instance of `T`, using its [`Arbitrary`] implementation.
    pub fn any<T: Arbitrary>(&mut self, label: &str) -> T {
        self.any_of(label, T::arbitrary())
    }

    /// Create an instance of type, using the supplied generator.
    pub fn any_of<G: Generator>(&mut self, label: &str, g: G) -> G::Item {
        self.as_raw().any_of(label, g, None)
    }

    /// Choose an element from a slice.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn choose<'values, T: Debug>(
        &mut self,
        label: &str,
        values: &'values [T],
    ) -> Option<(&'values T, usize)> {
        self.as_raw().choose(label, None, values)
    }

    /// Choose an element that satisfies the predicate from a slice.
    ///
    /// `choose_where` is similar to [`Generator::filter`]: it will return `None`
    /// if unable to choose value that satisfies the predicate in some number of tries.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn choose_where<'values, T: Debug>(
        &mut self,
        label: &str,
        values: &'values [T],
        pred: impl Fn(&T) -> bool,
    ) -> Option<(&'values T, usize)> {
        self.as_raw().choose_where(label, None, values, pred)
    }

    /// Choose a mutable element from a slice.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn choose_mut<'values, T: Debug>(
        &mut self,
        label: &str,
        values: &'values mut [T],
    ) -> Option<(&'values mut T, usize)> {
        self.as_raw().choose_mut(label, None, values)
    }

    /// Choose a mutable element that satisfies the predicate from a slice.
    ///
    /// `choose_mut_where` is similar to [`Generator::filter`]: it will return `None`
    /// if unable to choose value that satisfies the predicate in some number of tries.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn choose_mut_where<'values, T: Debug>(
        &mut self,
        label: &str,
        values: &'values mut [T],
        pred: impl Fn(&T) -> bool,
    ) -> Option<(&'values mut T, usize)> {
        self.as_raw().choose_mut_where(label, None, values, pred)
    }

    /// Try to generate `Some` value.
    ///
    /// `find` is similar to [`Generator::filter`]: it will return `None`
    /// if unable to generate `Some` in some number of tries.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn find<T: Debug>(
        &mut self,
        label: &str,
        func: impl Fn(&mut Self) -> Option<T>,
    ) -> Option<T> {
        let res = SourceRaw::find_impl(self, label, None, |src, _example| func(src));
        self.log_value(label, &res);
        res
    }

    /// Evaluate property in a sub-source to delimit several property checks inside a single [`Env::check`].
    #[cfg(test)]
    pub(crate) fn scope(&mut self, label: &str, prop: impl FnOnce(&mut Self)) {
        let mut src = Scope::new(self, label, "", false, ScopeKind::Plain, true);
        prop(&mut src);
    }

    /// Maybe proceed.
    pub fn maybe<T>(&mut self, label: &str, body: impl FnOnce(&mut Self) -> T) -> Option<T> {
        SourceRaw::maybe_impl(self, label, None, body)
    }

    /// Select a variant to proceed, where all variants are applicable.
    ///
    /// # Panics
    ///
    /// `select` panics when the variants slice is empty.
    pub fn select<T>(
        &mut self,
        label: &str,
        variants: &[&'static str],
        body: impl FnOnce(&mut Self, &str, usize) -> T,
    ) -> T {
        SourceRaw::select_impl(self, label, None, variants.len(), |ix| variants[ix], body)
    }

    /// Repeat `step` a number of times.
    pub fn repeat(&mut self, label: &str, step: impl FnMut(&mut Self) -> Effect) {
        self.repeat_n(label, .., step);
    }

    /// Repeat `step` specified number of times.
    pub fn repeat_n(
        &mut self,
        label: &str,
        n: impl RangeBounds<usize>,
        mut step: impl FnMut(&mut Self) -> Effect,
    ) {
        let res = SourceRaw::repeat_impl(
            self,
            label,
            Option::<core::iter::Empty<()>>::None,
            n,
            |_| (),
            |_v, src, _e| step(src),
        );
        res.assume_some_msg(UNABLE_PERFORM_MINIMUM_REPEAT);
    }

    /// Repeatedly select and execute some number of `step` variants that result in [`Effect::Success`].
    ///
    /// # Panics
    ///
    /// `repeat_select` panics when the variants slice is empty.
    // TODO: remove if not required for cover
    pub fn repeat_select(
        &mut self,
        label: &str,
        variants: &[&'static str],
        step: impl FnMut(&mut Self, &str, usize) -> Effect,
    ) {
        Self::repeat_select_n(self, label, .., variants, step);
    }

    /// Repeatedly select and execute specified number of `step` variants that result in [`Effect::Success`].
    ///
    /// # Panics
    ///
    /// `repeat_select_n` panics when the variants slice is empty.
    pub fn repeat_select_n(
        &mut self,
        label: &str,
        n_steps: impl RangeBounds<usize>,
        variants: &[&'static str],
        step: impl FnMut(&mut Self, &str, usize) -> Effect,
    ) {
        let ok = SourceRaw::repeat_select_impl(
            self,
            label,
            Option::<core::iter::Empty<usize>>::None,
            n_steps,
            variants.len(),
            |ix| variants[ix],
            step,
        );
        if !ok {
            __panic_assume(UNABLE_PERFORM_MINIMUM_REPEAT);
        }
    }

    /// Output properly indented debug formatting of value.
    pub fn log_value(&self, label: &str, v: &impl Debug) {
        self.as_ref().log_value(label, v);
    }

    /// Determine whether debug output should be used for a test case.
    ///
    /// Wrapping calls to [`println`], [`eprintln`], or [`dbg`]
    /// with a `should_log` check ensures that test debug output only contains information relevant
    /// to the failing test case. Additionally, omitting unnecessary debug output can significantly
    /// speed up the test.
    ///
    /// Alternatively, you can use helper macros [`vprintln`](crate::vprintln),
    /// [`veprintln`](crate::veprintln) or [`vdbg`](crate::vdbg) that encapsulate the `should_log` check.
    #[must_use]
    pub fn should_log(&self) -> bool {
        self.as_ref().should_log()
    }

    /// Require that condition is true at least once, when running in coverage mode.
    pub fn cover(&mut self, condition: &str, value: bool) {
        self.cover_all(&[(condition, value)]);
    }

    /// Require that all conditions are simultaneously true at least once, when running in coverage mode.
    pub fn cover_all(&mut self, conditions: &[(&str, bool)]) {
        self.as_mut().cover_all(conditions);
    }

    /// Require that at least one condition is true at least once, when running in coverage mode.
    pub fn cover_any(&mut self, conditions: &[(&str, bool)]) {
        self.as_mut().cover_any(conditions);
    }

    /// Get a reference to [`SourceRaw`].
    pub fn as_raw(&mut self) -> &mut SourceRaw<'env> {
        &mut self.raw
    }
}

impl<'env> Source<'env> {
    #[doc(hidden)]
    pub fn __new(env: &'env mut Env) -> Self {
        Self::new(env)
    }
}

/// Lower-level variant of [`Source`], primarily for use in [`Generator`] implementations.
#[derive(Debug)]
pub struct SourceRaw<'env> {
    env: &'env mut Env,
}

impl AsRef<Env> for SourceRaw<'_> {
    fn as_ref(&self) -> &Env {
        self.env
    }
}

impl AsMut<Env> for SourceRaw<'_> {
    fn as_mut(&mut self) -> &mut Env {
        self.env
    }
}

impl SourceRaw<'_> {
    /// Create an instance of `T`, using its [`Arbitrary`] implementation.
    pub fn any<T: Arbitrary>(&mut self, label: &str, example: Option<&T>) -> T {
        self.any_of(label, T::arbitrary(), example)
    }

    /// Create an instance of type, using the supplied generator.
    pub fn any_of<G: Generator>(
        &mut self,
        label: &str,
        g: G,
        example: Option<&G::Item>,
    ) -> G::Item {
        let v = {
            let mut src = Scope::new_plain(self, label, core::any::type_name::<G::Item>());
            g.next(&mut src, example)
        };
        self.log_value(label, &v);
        v
    }

    /// Choose an element from a slice.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn choose<'values, T: Debug>(
        &mut self,
        label: &str,
        example_index: Option<usize>,
        values: &'values [T],
    ) -> Option<(&'values T, usize)> {
        let ix = self.__choose_index(label, example_index, values.len());
        let v = ix.map(|ix| (&values[ix], ix));
        self.log_value(label, &v);
        v
    }

    /// Choose an element that satisfies the predicate from a slice.
    ///
    /// `choose_where` is similar to [`Generator::filter`]: it will return `None`
    /// if unable to choose value that satisfies the predicate in some number of tries.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn choose_where<'values, T: Debug>(
        &mut self,
        label: &str,
        example_index: Option<usize>,
        values: &'values [T],
        pred: impl Fn(&T) -> bool,
    ) -> Option<(&'values T, usize)> {
        let ix =
            self.__choose_index_where(label, example_index, values.len(), |i| pred(&values[i]));
        let v = ix.map(|ix| (&values[ix], ix));
        self.log_value(label, &v);
        v
    }

    /// Choose a mutable element from a slice.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn choose_mut<'values, T: Debug>(
        &mut self,
        label: &str,
        example_index: Option<usize>,
        values: &'values mut [T],
    ) -> Option<(&'values mut T, usize)> {
        let ix = self.__choose_index(label, example_index, values.len());
        let v = ix.map(|ix| (&mut values[ix], ix));
        self.log_value(label, &v);
        v
    }

    /// Choose a mutable element that satisfies the predicate from a slice.
    ///
    /// `choose_mut_where` is similar to [`Generator::filter`]: it will return `None`
    /// if unable to choose value that satisfies the predicate in some number of tries.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn choose_mut_where<'values, T: Debug>(
        &mut self,
        label: &str,
        example_index: Option<usize>,
        values: &'values mut [T],
        pred: impl Fn(&T) -> bool,
    ) -> Option<(&'values mut T, usize)> {
        let ix =
            self.__choose_index_where(label, example_index, values.len(), |i| pred(&values[i]));
        let v = ix.map(|ix| (&mut values[ix], ix));
        self.log_value(label, &v);
        v
    }

    #[doc(hidden)]
    pub fn __choose_index(
        &mut self,
        label: &str,
        example_index: Option<usize>,
        n: usize,
    ) -> Option<usize> {
        let mut src = Scope::new_plain(self, label, "<value-index>");
        let ix = (n != 0).then(|| src.as_mut().choose_index(n, example_index, Tweak::None));
        src.log_return(ix)
    }

    #[doc(hidden)]
    pub fn __choose_index_where(
        &mut self,
        label: &str,
        example_index: Option<usize>,
        n: usize,
        pred: impl Fn(usize) -> bool,
    ) -> Option<usize> {
        // TODO: this scope is not really required, but this way we can have tracing be consistent with `choose_index`
        let mut src = Scope::new_plain(self, label, "<value-index>");
        // TODO: bump the max repeat failures count to length of values?
        let ix: Option<Option<usize>> = Self::find_impl(
            &mut *src,
            label,
            example_index.map(Option::Some).as_ref(),
            |src, example| {
                if n == 0 {
                    // Short-circuit the `find`.
                    Some(None)
                } else {
                    let ix = src
                        .env
                        .choose_index(n, example.copied().flatten(), Tweak::None);
                    pred(ix).then_some(Some(ix))
                }
            },
        );
        src.log_return(ix.flatten())
    }

    /// Try to generate `Some` value.
    ///
    /// `find` is similar to [`Generator::filter`]: it will return `None`
    /// if unable to generate `Some` in some number of tries.
    ///
    /// Use [`OptionExt::assume_some`] if you can't handle the `None` value gracefully.
    pub fn find<T: Debug>(
        &mut self,
        label: &str,
        example: Option<&T>,
        func: impl Fn(&mut Self, Option<&T>) -> Option<T>,
    ) -> Option<T> {
        let res = Self::find_impl(self, label, example, func);
        self.log_value(label, &res);
        res
    }

    fn find_impl<T: Debug, Src: AsRef<Env> + AsMut<Env>>(
        src: &mut Src,
        label: &str,
        example: Option<&T>,
        next: impl Fn(&mut Src, Option<&T>) -> Option<T>,
    ) -> Option<T> {
        // TODO: use tape checkpoints instead of repeat? probably would need a special scope kind for better matching
        let res = Self::repeat_impl(
            src,
            label,
            example.map(|_| example.into_iter()),
            1..=1,
            |_n| None,
            |v, src, example| {
                *v = next(src, example);
                if v.is_some() {
                    Effect::Success
                } else {
                    Effect::Noop
                }
            },
        );
        res.map(|res| res.expect("internal error: find sequence must construct Some"))
    }

    /// Maybe proceed.
    pub fn maybe<T>(
        &mut self,
        label: &str,
        example: Option<bool>,
        body: impl FnOnce(&mut Self) -> T,
    ) -> Option<T> {
        Self::maybe_impl(self, label, example, body)
    }

    fn maybe_impl<S: AsRef<Env> + AsMut<Env>, T>(
        src: &mut S,
        label: &str,
        example: Option<bool>,
        body: impl FnOnce(&mut S) -> T,
    ) -> Option<T> {
        let go = {
            let mut src = Scope::new(src, label, "<maybe>", true, ScopeKind::RepeatSize, false);
            let size = src
                .as_mut()
                .choose_size(SizeRange::new_raw(0, 1), example.map(usize::from));
            let go = size != 0;
            src.log_return(go)
        };
        go.then(|| {
            let mut scope = Scope::new(src, label, "", false, ScopeKind::RepeatElement, true);
            body(&mut scope)
        })
    }

    /// Select a variant to proceed, where all variants are applicable.
    pub fn select<L: AsRef<str> + Debug, T>(
        &mut self,
        label: &str,
        example_index: Option<usize>,
        variants: NonZero<usize>,
        variant_label: impl Fn(usize) -> L,
        body: impl FnOnce(&mut Self, &str, usize) -> T,
    ) -> T {
        Self::select_impl(
            self,
            label,
            example_index,
            variants.get(),
            variant_label,
            body,
        )
    }

    #[track_caller]
    fn select_impl<S: AsRef<Env> + AsMut<Env>, L: AsRef<str> + Debug, T>(
        src: &mut S,
        label: &str,
        example_index: Option<usize>,
        variants: usize,
        variant_label: impl Fn(usize) -> L,
        body: impl FnOnce(&mut S, &str, usize) -> T,
    ) -> T {
        assert!(variants > 0, "no variants provided");
        let (variant, ix) = {
            let mut src = Scope::new(
                src,
                label,
                "<variant-index>",
                false,
                ScopeKind::SelectIndex,
                false,
            );
            let ix = src
                .as_mut()
                .choose_index(variants, example_index, Tweak::None);
            let variant = variant_label(ix);
            src.log_return((variant, ix))
        };
        let mut scope = Scope::new_select_variant(src, label, variant.as_ref(), ix);
        body(&mut scope, variant.as_ref(), ix)
    }

    /// Repeat `step` several times, after executing the `setup` once.
    pub fn repeat<S: Debug, T>(
        &mut self,
        label: &str,
        example: Option<impl ExactSizeIterator<Item = T>>,
        n_steps: impl RangeBounds<usize>,
        setup: impl FnOnce(usize) -> S,
        step: impl FnMut(&mut S, &mut Self, Option<T>) -> Effect,
    ) -> Option<S> {
        let res = Self::repeat_impl(self, label, example, n_steps, setup, step);
        self.log_value(label, &res); // TODO: do we need this?
        res
    }

    #[track_caller]
    fn repeat_impl<S, T, Src: AsRef<Env> + AsMut<Env>>(
        src: &mut Src,
        label: &str,
        mut example: Option<impl ExactSizeIterator<Item = T>>,
        n_steps: impl RangeBounds<usize>,
        setup: impl FnOnce(usize) -> S,
        mut step: impl FnMut(&mut S, &mut Src, Option<T>) -> Effect,
    ) -> Option<S> {
        let r = SizeRange::new(n_steps);
        let (must, extra) = {
            let mut src = Scope::new(src, label, "<size>", true, ScopeKind::RepeatSize, false);
            let total = src
                .as_mut()
                .choose_size(r, example.as_ref().map(ExactSizeIterator::len));
            let (must, extra) = (r.min, total - r.min);
            src.log_return((must as u32, extra as u32))
        };
        let mut state = setup((must + extra) as usize);
        let mut done = 0;
        let mut reject_conseq_soft = 0;
        let mut reject_conseq_hard = 0;
        let mut step_version = 1; // 0 means inherit the version, we don't want to
        let mut enum_mode = false;
        let mut counter = 0;
        while done < must + extra {
            let mut src = Scope::new_repeat_element(src, label, counter, step_version, enum_mode);
            // Note: we advance the example each time, otherwise example might give us e.g. Noop infinitely.
            let example = example.as_mut().and_then(Iterator::next);
            let effect = step(&mut state, &mut src, example);
            if effect == Effect::Success {
                done += 1;
                reject_conseq_soft = 0;
                reject_conseq_hard = 0;
                // Go back to swarm selection if enumeration got us unstuck.
                if enum_mode {
                    enum_mode = false;
                    step_version += 1;
                }
            } else {
                src.mark_effect(label, counter as usize, effect);
                reject_conseq_hard += 1;
                if reject_conseq_hard == REPEAT_REJECT_CONSEQ_HARD_MAX {
                    // Too many total rejects means we should abandon the repeat.
                    return (done >= must).then_some(state);
                }
                reject_conseq_soft += 1;
                if enum_mode {
                    // Continue the enumeration.
                    step_version += 1;
                } else if reject_conseq_soft == REPEAT_REJECT_CONSEQ_SOFT_MAX {
                    // Too many consecutive rejects means we should switch to the enumeration mode.
                    reject_conseq_soft = 0;
                    step_version += 1;
                    enum_mode = true;
                }
            }
            counter += 1;
        }
        Some(state)
    }

    /// Repeatedly select and execute specified number of `step` variants that result in [`Effect::Success`].
    pub fn repeat_select<L: AsRef<str> + Debug>(
        &mut self,
        label: &str,
        example_indices: Option<impl ExactSizeIterator<Item = usize>>,
        n_steps: impl RangeBounds<usize>,
        variants: NonZero<usize>,
        variant_label: impl Fn(usize) -> L,
        step: impl FnMut(&mut Self, &str, usize) -> Effect,
    ) -> bool {
        Self::repeat_select_impl(
            self,
            label,
            example_indices,
            n_steps,
            variants.get(),
            variant_label,
            step,
        )
    }

    // We don't track/return total effect, since we don't want people
    // to build `repeat_select` hierarchies.
    // So this is really just a `repeat` + `select`rolled into one, for now.
    #[track_caller]
    fn repeat_select_impl<S: AsRef<Env> + AsMut<Env>, L: AsRef<str> + Debug>(
        src: &mut S,
        label: &str,
        example_indices: Option<impl ExactSizeIterator<Item = usize>>,
        n_steps: impl RangeBounds<usize>,
        variants: usize,
        variant_label: impl Fn(usize) -> L,
        mut step: impl FnMut(&mut S, &str, usize) -> Effect,
    ) -> bool {
        assert!(variants > 0, "no variants provided");
        // Note: we use the same base label both in repeat and select.
        let res = Self::repeat_impl(
            src,
            label,
            example_indices,
            n_steps,
            |_n| (),
            |(), src, example_index| {
                // TODO: ensure that repeat gives us the right index
                Self::select_impl(
                    src,
                    label,
                    example_index,
                    variants,
                    &variant_label,
                    &mut step,
                )
            },
        );
        res.is_some()
    }

    /// Output properly indented debug formatting of value.
    pub fn log_value(&self, label: &str, v: &impl Debug) {
        self.env.log_value(label, v);
    }

    /// Determine whether debug output should be used for a test case.
    ///
    /// Wrapping calls to [`println`], [`eprintln`], or [`dbg`]
    /// with a `should_log` check ensures that test debug output only contains information relevant
    /// to the failing test case. Additionally, omitting unnecessary debug output can significantly
    /// speed up the test.
    ///
    /// Alternatively, you can use helper macros [`vprintln`](crate::vprintln),
    /// [`veprintln`](crate::veprintln) or [`vdbg`](crate::vdbg) that encapsulate the `should_log` check.
    #[must_use]
    pub fn should_log(&self) -> bool {
        self.env.should_log()
    }

    // Hack to don't try to mutate choices that will not matter anyway.
    pub(crate) fn mark_next_choice_forced(&mut self) {
        self.env.mark_next_choice_forced();
    }

    #[cfg(test)]
    pub(crate) fn derived_oneshot_env(&self, tape: Tape) -> Env {
        // Note: the seed is randomized.
        Env::custom()
            .with_rng_budget(self.env.budget())
            .with_rng_tape(tape)
            .with_check_iters(1)
            .with_reduce_time(Duration::ZERO)
            .env(false)
    }
}
