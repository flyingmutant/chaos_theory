// Shared derive cases used by runtime and expansion tests.

#[derive(Debug, Clone, PartialEq, Eq, chaos_theory::Arbitrary)]
struct Point {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, chaos_theory::Arbitrary)]
struct Triple(u8, bool, Option<i16>);

#[derive(Debug, Clone, PartialEq, Eq, chaos_theory::Arbitrary)]
struct Marker;

#[derive(Debug, Clone, PartialEq, Eq, Arbitrary)]
struct Imported {
    v: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, chaos_theory::Arbitrary)]
struct Wrapper<T> {
    inner: T,
}

#[derive(Debug, Clone, PartialEq, Eq, chaos_theory::Arbitrary)]
enum Action<T> {
    Reset,
    Set(T),
    Shift { by: i16 },
}
