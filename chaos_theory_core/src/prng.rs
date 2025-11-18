#![expect(dead_code)]

trait Arbitrary {}

trait Generator {
    type Item;

    fn next(&self, src: &mut Source) -> Self::Item;
    // example? external value decomposition? but decomposition can be different (it should mirror serialization)!
}
// just a Fn(&mut Source)?

struct Source {}

// labels?
// scopes? optional! only for shrinking; OK if only in built-in-primitives they are good, and in user code they don't really exist
// - generated values will have good scopes due to generators being nice
// - control flow will more or less have no good scopes due to users being lazy
// inject examples?

// can we infer labels?
// can we infer repeat/select scope markup?

impl Source {
    // any::<bool> needs to work in complex boolean expressions (foo && src.any("bar") etc.)

    fn any<T: Arbitrary>(&mut self) -> T {
        todo!()
    }

    fn any_of<G: Generator>(&mut self, _g: G) -> G::Item {
        todo!()
    }

    fn choose<T>(&mut self, _elems: &[T]) -> &T {
        // any_of(one_of)
        todo!()
    }
    // -> Option<&T>?

    fn choose_where<T>(&mut self, _elems: &[T], _pred: impl Fn(&T) -> bool) -> &T {
        // any_of(one_of.filter)
        todo!()
    }
    // find?

    // repeat? any("count") loop (with custom += 1), any("continue") loop, plus rejection-sampling
    // select?
    // split?
    // permute?

    // size/index/value internal primitives

    // cover?
}

// orthogonal scopes & generation
// generator can ask "where am i"
// you can provide scopes but not required to

// orthogonal "draw" & "serialize"
