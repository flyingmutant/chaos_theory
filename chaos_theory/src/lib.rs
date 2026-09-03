// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/*!
`chaos_theory` is a modern property-based testing and structure-aware fuzzing library.

You drive tests using [`Source`] to get structured pseudo-random values and control flow,
`chaos_theory` ensures that this exploration is efficient and any failures found
are automatically minimized.

# Quickstart

```rust
use chaos_theory::check;

#[test]
fn sort_strings() {
    check(|src| {
        let mut strings: Vec<String> = src.any("strings");
        strings.sort();
        assert!(strings.is_sorted(), "unsorted after sort: {strings:?}");
    });
}
```

More complete examples:

- [`parse_date` function test](crate::_examples::parse_date)
- [`Queue` state machine test](crate::_examples::queue)

When a failure happens, `chaos_theory` prints a `CHAOS_THEORY_REPLAY=...` string
you can use to reproduce the case.

# Highlights

- Property testing and structure-aware fuzzing in one library
- Efficient state space exploration:
  - bias towards small values and edge cases
  - structural mutations and crossover
  - example-guided generation
  - built-in swarm testing
- Macro-free, immediate-mode API: generate values and control flow as the test runs
- Zero unsafe code, zero required dependencies and `no_std`-compatible

# Documentation

- [Guide](crate::_docs::guide)
- [FAQ](crate::_docs::faq)
- [Changelog](crate::_docs::changelog)
*/

#![cfg_attr(all(test, feature = "_bench"), feature(test))]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), allow(dead_code))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#[cfg(all(not(feature = "std"), not(feature = "no_std")))]
compile_error!("Enable `feature = \"no_std\"` when building `chaos_theory` without `std`.");

extern crate alloc;
#[cfg(feature = "derive")]
extern crate self as chaos_theory;
#[cfg(all(test, feature = "_bench"))]
extern crate test;

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use chaos_theory_derive::Arbitrary;

mod base64;
mod config;
mod cover;
mod distrib;
mod env;
mod generator;
mod hash;
mod hash_identity;
mod jumphash;
mod libfuzzer;
mod math;
mod num;
mod permute;
mod rand;
mod range;
#[cfg(feature = "std")]
mod reduce;
mod source;
mod tape;
mod tape_event;
mod tape_mutate;
mod tape_mutate_crossover;
#[cfg(feature = "std")]
mod tape_reduce;
mod tape_validate;
#[cfg(any(test, all(not(feature = "std"), feature = "derive")))]
mod tests;
#[cfg(test)]
mod tests_shrinking_challenge;
mod unwind;
mod util;
mod varint;

#[cfg(doc)]
pub mod _docs;
#[cfg(doc)]
pub mod _examples;

#[cfg(feature = "bstr")]
mod make_bstr;
#[cfg(feature = "bytes")]
mod make_bytes;
mod make_cell;
mod make_char;
mod make_collection;
mod make_combine;
mod make_core;
#[cfg(feature = "ecow")]
mod make_ecow;
#[cfg(feature = "either")]
mod make_either;
mod make_float;
#[cfg(feature = "hashbrown")]
mod make_hashbrown;
#[cfg(feature = "indexmap")]
mod make_indexmap;
mod make_integer;
#[cfg(feature = "jiff")]
mod make_jiff;
mod make_net;
#[cfg(feature = "ordered_float")]
mod make_ordered_float;
#[cfg(feature = "ordermap")]
mod make_ordermap;
#[cfg(feature = "regex")]
mod make_regex;
#[cfg(feature = "serde_json")]
mod make_serde_json;
mod make_special;
mod make_string;
#[cfg(feature = "std")]
mod make_sync;
mod make_time;
#[cfg(feature = "tinyvec")]
mod make_tinyvec;
mod make_tuple;
#[cfg(feature = "uuid")]
mod make_uuid;
#[cfg(feature = "jiff")]
mod time_zone_names {
    include!("../gen/time_zone_names.rs");
}

pub use config::Config;
pub use env::{Effect, Env};
pub use generator::{Arbitrary, Gen, GenShared, Generator};
pub use num::{Float, Int, Ranged, Unsigned};
pub use source::{Source, SourceEx};
pub use util::{MaybeOwned, OptionExt, should_log};

#[doc(hidden)]
pub mod __private {
    #[cfg(feature = "std")]
    pub use crate::env::FuzzState;
    pub use crate::unwind::{ASSUME_FAILED_PREFIX, catch_silent, panic_assume, panic_message};
}

/// Collection of built-in generator implementations.
pub mod make {
    pub use crate::make_cell::{cell, once_cell, ref_cell, unsafe_cell};
    pub use crate::make_char::{byte_ascii, char_ascii};
    pub use crate::make_collection::{
        array, btree_map, btree_map_n, btree_set, btree_set_n, slice, slice_n, vec, vec_n,
    };
    #[cfg(feature = "std")]
    pub use crate::make_collection::{hash_map, hash_map_n, hash_set, hash_set_n};
    pub use crate::make_combine::{
        from_fn, from_fn_find, from_next, from_next_find, just, mix_of, mix_of_owned, one_of,
        one_of_owned,
    };
    pub use crate::make_core::{err, none, ok, option, result, some};
    pub use crate::make_float::float_in;
    pub use crate::make_integer::int_in;
    #[cfg(feature = "regex")]
    pub use crate::make_regex::{
        byte_slice_matching, bytes_matching, cstring_matching, cstring_slice_matching,
        string_matching, string_slice_matching,
    };
    pub use crate::make_special::{index, size, token, try_index};
    pub use crate::make_string::{
        cstring, cstring_n, cstring_slice, cstring_slice_n, string, string_n, string_slice,
        string_slice_n,
    };
    #[cfg(feature = "std")]
    pub use crate::make_sync::{barrier, mpsc_sync_channel, mutex, once_lock, rw_lock};
    pub use crate::make_time::duration_in;
    #[cfg(feature = "std")]
    pub use crate::make_time::system_time_in;

    use crate::{Arbitrary, Generator};

    /// Create a generator of `T`, using its [`Arbitrary`] implementation.
    ///
    /// This is equivalent to `<T as Arbitrary>::arbitrary()`, but can sometimes be a bit more concise.
    pub fn arbitrary<T: Arbitrary>() -> impl Generator<Item = T> {
        T::arbitrary()
    }

    #[cfg(feature = "bstr")]
    /// [`bstr`](https://docs.rs/bstr) generators.
    pub mod bstr {
        pub use crate::make_bstr::{bstring, bstring_n};
    }

    #[cfg(feature = "bytes")]
    /// [`bytes`](https://docs.rs/bytes) generators.
    pub mod bytes {
        pub use crate::make_bytes::{bytes, bytes_mut, bytes_mut_n, bytes_n};
    }

    #[cfg(feature = "hashbrown")]
    /// [`hashbrown`](https://docs.rs/hashbrown) generators.
    pub mod hashbrown {
        pub use crate::make_hashbrown::{hash_map, hash_map_n, hash_set, hash_set_n};
    }

    #[cfg(feature = "ecow")]
    /// [`ecow`](https://docs.rs/ecow) generators.
    pub mod ecow {
        pub use crate::make_ecow::{
            eco_bytes, eco_bytes_n, eco_string, eco_string_n, eco_vec, eco_vec_n,
        };
    }

    #[cfg(feature = "either")]
    /// [`either`](https://docs.rs/either) generators.
    pub mod either {
        pub use crate::make_either::{either, left, right};
    }

    #[cfg(feature = "indexmap")]
    /// [`indexmap`](https://docs.rs/indexmap) generators.
    pub mod indexmap {
        pub use crate::make_indexmap::{index_map, index_map_n, index_set, index_set_n};
    }

    #[cfg(feature = "jiff")]
    /// [`jiff`](https://docs.rs/jiff) generators.
    pub mod jiff {
        pub use crate::make_jiff::{signed_duration_in, timestamp_in};
    }

    #[cfg(feature = "ordermap")]
    /// [`ordermap`](https://docs.rs/ordermap) generators.
    pub mod ordermap {
        pub use crate::make_ordermap::{order_map, order_map_n, order_set, order_set_n};
    }

    #[cfg(feature = "serde_json")]
    /// [`serde_json`](https://docs.rs/serde_json) generators.
    pub mod serde_json {
        pub use crate::make_serde_json::{json_number, json_object, json_value};
    }

    #[cfg(feature = "ordered_float")]
    /// [`ordered_float`](https://docs.rs/ordered-float) generators.
    pub mod ordered_float {
        pub use crate::make_ordered_float::{not_nan, not_nan_in, ordered_float, ordered_float_in};
    }

    #[cfg(feature = "tinyvec")]
    /// [`tinyvec`](https://docs.rs/tinyvec) generators.
    pub mod tinyvec {
        pub use crate::make_tinyvec::{array_vec, array_vec_n, tiny_vec, tiny_vec_n};
    }

    #[cfg(feature = "uuid")]
    /// [`uuid`](https://docs.rs/uuid) generators.
    pub mod uuid {
        pub use crate::make_uuid::uuid_v4;
    }
}

/// Check that property holds (does not panic).
///
/// To customize the `check` behavior, use [`Env::builder`] and [`Env::check`].
/// Environment variables described in [`Config::build`] are used as configuration fallbacks.
#[cfg(feature = "std")]
pub fn check(prop: impl Fn(&mut Source)) {
    Env::builder().with_env_vars().build().check(prop);
}

/// Advance the deterministic `no_std` seed sequence by `steps`.
///
/// This affects default seeding in `no_std`, such as [`Env::new`].
/// On targets without 32-bit atomics, the `no_std` fallback seed is fixed and this has no effect.
#[cfg(not(feature = "std"))]
pub fn jump_seed_sequence(steps: u64) {
    rand::jump_seed_sequence(steps);
}

#[cfg(feature = "std")]
pub(crate) type Set<K> = std::collections::HashSet<K, hash::FxBuildHasher>;
#[cfg(not(feature = "std"))]
pub(crate) type Set<K> = alloc::collections::BTreeSet<K>;

#[cfg(feature = "std")]
pub(crate) type Map<K, V> = std::collections::HashMap<K, V, hash::FxBuildHasher>;
#[cfg(not(feature = "std"))]
pub(crate) type Map<K, V> = alloc::collections::BTreeMap<K, V>;
