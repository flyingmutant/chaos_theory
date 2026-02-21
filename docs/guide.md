# Guide

This is a short guide to chaos_theory basics. It covers how to write properties with
`Source`, with a focus on the two core building blocks after `any`: `repeat` and `select`.

## The Shape Of A Property

Property tests in chaos_theory look like this:

```rust
use chaos_theory::check;

check(|src| {
    // generate structured values
    // drive the system under test
    // assert invariants
});
```

`Source` gives you structured randomness. Your job is to explore the system using it.

## Working With `Source`

The basic operations:

- `any` / `any_of` generate values from `Arbitrary` or from a generator.
- `choose` selects an element from a slice.
- `select` chooses a labeled variant and runs a branch.
- `repeat` repeats a step, using `Effect` to report what happened.
- `maybe` and `find` are for optional steps and recoverable failures.

Labels matter. Use short, stable labels (like `"action"` or `"key"`). They are
not required for chaos_theory to work, but they make replay output and
reproduction steps readable. With good labels, the failing case description is
often enough to spot the issue immediately.

## `select`: Variants With Meaning

`select` is how you define structured choices:

```rust
src.select("action", &["insert", "remove", "get"], |src, action, _ix| {
    match action {
        "insert" => { /* ... */ }
        "remove" => { /* ... */ }
        "get" => { /* ... */ }
        _ => unreachable!(),
    }
});
```

You should not encode a variant choice as `any::<u8>()` or a random number. Use
`select` so replay and minimization can preserve the variant choice when necessary.

## `repeat`: Exploration Over Time

`repeat` is the right way to explore sequences:

```rust
use chaos_theory::Effect;

src.repeat("step", |src| {
    // perform one step
    Effect::Success
});
```

`Effect` matters:

- `Success`: useful work was done.
- `Change`: state may have changed, but no clear progress.
- `Noop`: nothing happened to the system (example: action was non-applicable).

Honest `Effect` values make exploration and minimization much more efficient.

### Common Anti-Pattern: Manual Random Loops

Don't do this:

```rust
let n: usize = src.any("n");
for _ in 0..n {
    /* use src here */
}
```

Use `repeat` instead. `repeat` is structured and minimizes well, while manual
random loops are opaque and minimize poorly.

Another version of the same issue is:

```rust
let do_it: bool = src.any("do_it");
if do_it { /* use src here */ }
```

Prefer `maybe` or `select` so the execution shape is tracked structurally.

## Stateful Testing (State Machines)

chaos_theory does not have a special API for state machines. The normal API is
already the "advanced" mode.

The most common pattern is:

1. Build the system under test and a reference model.
2. `repeat` a step that selects and applies an action.
3. Assert invariants or compare against the model.

Example shape:

```rust
src.repeat("step", |src| {
    src.select("action", &["insert", "remove", "get"], |src, action, _| {
        // apply action to SUT
        // apply action to model
        // assert invariants
        // return Effect for the chosen action
    })
});
```

Nested `repeat` and `select` are normal and encouraged for complex stateful systems.

## Filtering And Validity

If you need to reject invalid values, prefer recoverable filtering:

- `Generator::filter` returns `Option` so you can handle failure without panicking.
- `filter_assume` and `assume!` mark the whole test case as invalid when the condition fails.

Too many invalid cases will make `check` fail early because it cannot generate enough valid tests.

## Debugging Output

Use `should_log`, `vdbg!`, and `vprintln!` so output appears only for the
failing case. It keeps tests fast and logs focused.

## Generators Are Optional

Most users never write custom generators. You can get far with:

- built-in generators in `make::*`,
- composing with `select`, `repeat`, and `any`,
- occasional use of `from_fn` if needed.

A derive macro for `Arbitrary` will make creating generators much easier in the future.
