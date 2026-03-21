// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Integration tests for `#[derive(chaos_theory::Arbitrary)]`.

#![cfg(feature = "derive")]

use chaos_theory::{Arbitrary, Source, check};

include!("../../testdata/derive_cases.rs");

#[derive(Debug)]
struct NotArbitrary;

fn assert_reconstruct<T>(src: &mut Source, example_label: &str, value_label: &str)
where
    T: chaos_theory::Arbitrary + core::fmt::Debug + PartialEq,
{
    let example: T = src.any(example_label);
    let value = src.as_raw().any(value_label, Some(&example));
    assert_eq!(value, example);
}

#[test]
fn derive_reconstructs_examples() {
    check(|src| {
        assert_reconstruct::<Point>(src, "point_example", "point_value");
        assert_reconstruct::<Triple>(src, "triple_example", "triple_value");
        assert_reconstruct::<Marker>(src, "marker_example", "marker_value");
        assert_reconstruct::<Imported>(src, "imported_example", "imported_value");
        assert_reconstruct::<Wrapper<u16>>(src, "wrapper_example", "wrapper_value");
        assert_reconstruct::<CustomPoint>(src, "custom_point_example", "custom_point_value");
        assert_reconstruct::<Action<u8>>(src, "action_example", "action_value");
        assert_reconstruct::<CustomAction<u8>>(src, "custom_action_example", "custom_action_value");
    });
}

#[test]
fn derive_uses_custom_field_generators() {
    check(|src| {
        let point: CustomPoint = src.any("custom_point");
        assert_eq!(point.x % 2, 1);

        let action: CustomAction<bool> = src.any("custom_action");
        match action {
            CustomAction::Reset => {}
            CustomAction::Set(value, _) => assert_eq!(value % 2, 1),
            CustomAction::Shift { by } => assert_eq!(by % 2, 0),
        }

        let wrapper: CustomWrapper<NotArbitrary> = src.any("custom_wrapper");
        assert!(wrapper.value.is_none());
    });
}
