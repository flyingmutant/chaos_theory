// Shared derive cases used by runtime and expansion tests.

fn odd_u8() -> impl chaos_theory::Generator<Item = u8> {
    chaos_theory::make::from_fn(|src, example| src.any::<u8>("base", example) | 1)
}

fn even_i16() -> impl chaos_theory::Generator<Item = i16> {
    chaos_theory::make::from_fn(|src, example| src.any::<i16>("base", example) & !1)
}

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
struct CustomPoint {
    #[chaos_theory(generator = odd_u8())]
    x: u8,
    y: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, chaos_theory::Arbitrary)]
enum Action<T> {
    Reset,
    Set(T),
    Shift { by: i16 },
}

#[derive(Debug, Clone, PartialEq, Eq, chaos_theory::Arbitrary)]
enum CustomAction<T> {
    Reset,
    Set(#[chaos_theory(generator = odd_u8())] u8, T),
    Shift {
        #[chaos_theory(generator = even_i16())]
        by: i16,
    },
}

#[derive(Debug, chaos_theory::Arbitrary)]
struct CustomWrapper<T: core::fmt::Debug> {
    #[chaos_theory(generator = chaos_theory::make::from_fn(|_src, _example| None))]
    value: Option<T>,
}
