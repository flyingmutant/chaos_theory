/*!
Language-independent `chaos_theory` core.
*/

#![no_std]
#![forbid(unsafe_code)]
#![expect(missing_docs)] // TODO

extern crate alloc;

#[cfg(test)]
extern crate std;

mod label;
mod tracer;

pub use tracer::{SpanId, Tracer};
