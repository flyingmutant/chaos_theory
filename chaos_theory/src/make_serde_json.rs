use alloc::string::String;
use core::num::NonZero;

use serde_json::{Map, Number, Value};

use crate::{
    Arbitrary, Effect, Generator, OptionExt as _, SourceRaw, UNABLE_GENERATE_UNIQUE, make,
    range::SizeRange,
};

#[cfg_attr(docsrs, doc(cfg(feature = "serde_json")))]
impl Arbitrary for Number {
    fn arbitrary() -> impl Generator<Item = Self> {
        json_number()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "serde_json")))]
impl Arbitrary for Map<String, Value> {
    fn arbitrary() -> impl Generator<Item = Self> {
        json_object(String::arbitrary(), Value::arbitrary())
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "serde_json")))]
impl Arbitrary for Value {
    fn arbitrary() -> impl Generator<Item = Self> {
        json_value()
    }
}

/// Create a [`serde_json::Number`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "serde_json")))]
pub fn json_number() -> impl Generator<Item = Number> {
    Number_ {}
}

/// Create a [`serde_json::Map<String, Value>`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "serde_json")))]
pub fn json_object(
    key: impl Generator<Item = String>,
    value: impl Generator<Item = Value>,
) -> impl Generator<Item = Map<String, Value>> {
    Object_ {
        key,
        value,
        size: SizeRange::new(..),
    }
}

/// Create a [`serde_json::Value`] generator.
#[cfg_attr(docsrs, doc(cfg(feature = "serde_json")))]
pub fn json_value() -> impl Generator<Item = Value> {
    Value_ {}
}

#[derive(Debug)]
struct Object_<GK, GV> {
    key: GK,
    value: GV,
    size: SizeRange,
}

impl<GK: Generator<Item = String>, GV: Generator<Item = Value>> Generator for Object_<GK, GV> {
    type Item = Map<String, Value>;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_seq = example.map(|e| e.iter());
        let res = src.repeat(
            "<object>",
            example_seq,
            self.size,
            Map::with_capacity,
            |v, src, example| {
                use serde_json::map::Entry::{Occupied, Vacant};
                let key = src.any_of("<key>", &self.key, example.map(|e| e.0));
                match v.entry(key) {
                    Occupied(_) => Effect::Noop,
                    Vacant(e) => {
                        let val = src.any_of("<value>", &self.value, example.map(|e| e.1));
                        e.insert(val);
                        Effect::Success
                    }
                }
            },
        );
        res.assume_some_msg(UNABLE_GENERATE_UNIQUE)
    }
}

#[derive(Debug)]
struct Number_ {}

impl Generator for Number_ {
    type Item = Number;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_index = example.and_then(|e| {
            if e.is_u64() {
                Some(0)
            } else if e.is_i64() {
                Some(1)
            } else if e.is_f64() {
                Some(2)
            } else {
                None
            }
        });

        let variants = &["uint", "int", "float"];
        let variants_num = NonZero::new(variants.len()).expect("internal error: no variants");
        src.select(
            "<number>",
            example_index,
            variants_num,
            |ix| variants[ix],
            |src, variant, _ix| match variant {
                "uint" => {
                    let example = example.and_then(Number::as_u64);
                    Number::from(make::arbitrary::<u64>().next(src, example.as_ref()))
                }
                "int" => {
                    let example = example.and_then(Number::as_i64);
                    Number::from(make::int_in_range(i64::MIN..=-1).next(src, example.as_ref()))
                }
                "float" => {
                    let example = example.and_then(Number::as_f64);
                    if let Some(example) = example {
                        debug_assert!(example.is_finite());
                    }
                    Number::from_f64(
                        make::float_in_range(f64::MIN..=f64::MAX).next(src, example.as_ref()),
                    )
                    .expect("internal error: can't construct JSON number from finite f64")
                }
                _ => unreachable!(),
            },
        )
    }
}

#[derive(Debug)]
struct Value_ {}

impl Generator for Value_ {
    type Item = Value;

    fn next(&self, src: &mut SourceRaw, example: Option<&Self::Item>) -> Self::Item {
        let example_index = example.map(|e| match e {
            Value::Null => 0,
            Value::Bool(_) => 1,
            Value::Number(_) => 2,
            Value::String(_) => 3,
            Value::Array(_) => 4,
            Value::Object(_) => 5,
        });

        let variants = &["Null", "Bool", "Number", "String", "Array", "Object"];
        let variants_num = NonZero::new(variants.len()).expect("internal error: no variants");
        src.select(
            "<value>",
            example_index,
            variants_num,
            |ix| variants[ix],
            |src, variant, _ix| match variant {
                "Null" => Value::Null,
                "Bool" => {
                    let example = match example {
                        Some(Value::Bool(v)) => Some(v),
                        _ => None,
                    };
                    let v = make::arbitrary().next(src, example);
                    Value::Bool(v)
                }
                "Number" => {
                    let example = match example {
                        Some(Value::Number(v)) => Some(v),
                        _ => None,
                    };
                    let v = json_number().next(src, example);
                    Value::Number(v)
                }
                "String" => {
                    let example = match example {
                        Some(Value::String(v)) => Some(v),
                        _ => None,
                    };
                    let v = make::arbitrary().next(src, example);
                    Value::String(v)
                }
                "Array" => {
                    let example = match example {
                        Some(Value::Array(v)) => Some(v),
                        _ => None,
                    };
                    let v = make::vec(json_value()).next(src, example);
                    Value::Array(v)
                }
                "Object" => {
                    let example = match example {
                        Some(Value::Object(v)) => Some(v),
                        _ => None,
                    };
                    let v = make::arbitrary().next(src, example);
                    Value::Object(v)
                }
                _ => unreachable!(),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, tests::prop_smoke};

    #[test]
    fn serde_json_smoke() {
        check(|src| {
            prop_smoke(src, "Number", Number::arbitrary());
            prop_smoke(src, "Map<String, Value>", Map::<String, Value>::arbitrary());
            prop_smoke(src, "Value", Value::arbitrary());
        });
    }
}
