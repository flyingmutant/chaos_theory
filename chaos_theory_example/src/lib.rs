#![expect(dead_code)]

enum ActionError {
    Precondition,
    Other,
}

type ActionResult = Result<(), ActionError>;

trait ActionOption {
    type Value;

    fn pre(self) -> Result<Self::Value, ActionError>;
    fn err(self) -> Result<Self::Value, ActionError>;
}

impl<T> ActionOption for Option<T> {
    type Value = T;

    fn pre(self) -> Result<Self::Value, ActionError> {
        self.ok_or(ActionError::Precondition)
    }

    fn err(self) -> Result<Self::Value, ActionError> {
        self.ok_or(ActionError::Other)
    }
}

#[cfg(test)]
mod tests {
    use actions::Action as _;
    use chaos_theory::{Effect, Source, check};

    use crate::{ActionError, ActionResult};

    mod api {
        pub fn create(_key: &str, _val: &str) {}
        pub fn update(_key: &str, _val: &str) {}
        pub fn delete(_key: &str) {}
    }

    mod actions {
        use chaos_theory::{Effect, Generator, Source, make};

        use crate::{ActionOption as _, ActionResult};

        use super::api;

        #[derive(Debug)]
        struct Pair {
            key: String,
            val: String,
        }

        #[derive(Default)]
        pub struct State {
            pairs: Vec<Pair>,
        }

        pub trait Action: Sized {
            fn new(src: &mut Source, state: &State) -> Option<Self>;
            fn apply(self, state: &mut State);

            fn run(src: &mut Source, state: &mut State) -> Effect {
                let Some(act) = Self::new(src, state) else {
                    return Effect::Noop;
                };
                act.apply(state);
                Effect::Success
            }
        }

        pub struct CreateAction {
            key: String,
            val: String,
        }

        impl Action for CreateAction {
            fn new(src: &mut Source, state: &State) -> Option<Self> {
                let key: String = src.any_of(
                    "key",
                    make::arbitrary().filter(|k| state.pairs.iter().all(|p| &p.key != k)),
                )?;
                let val: String = src.any("val");
                Some(Self { key, val })
            }

            fn apply(self, state: &mut State) {
                api::create(&self.key, &self.val);
                state.pairs.push(Pair {
                    key: self.key,
                    val: self.val,
                });
            }
        }

        pub struct UpdateAction {
            pair_ix: usize,
            val: String,
        }

        impl Action for UpdateAction {
            fn new(src: &mut Source, state: &State) -> Option<Self> {
                let (_, pair_ix) = src.choose("pair", &state.pairs)?;
                let val: String = src.any("val");
                Some(Self { pair_ix, val })
            }

            fn apply(self, state: &mut State) {
                let p = &mut state.pairs[self.pair_ix];
                api::update(&p.key, &self.val);
                p.val = self.val;
            }
        }

        pub struct DeleteAction {
            pair_ix: usize,
        }

        impl Action for DeleteAction {
            fn new(src: &mut Source, state: &State) -> Option<Self> {
                let (_, pair_ix) = src.choose("pair", &state.pairs)?;
                Some(Self { pair_ix })
            }

            fn apply(self, state: &mut State) {
                let p = &state.pairs[self.pair_ix];
                api::delete(&p.key);
                state.pairs.swap_remove(self.pair_ix);
            }
        }

        pub fn create(src: &mut Source, state: &mut State) -> ActionResult {
            let key: String = src
                .any_of(
                    "key",
                    make::arbitrary().filter(|k| state.pairs.iter().all(|p| &p.key != k)),
                )
                .pre()?;
            let val: String = src.any("val");
            api::create(&key, &val);
            state.pairs.push(Pair { key, val });
            Ok(())
        }

        pub fn update(src: &mut Source, state: &mut State) -> ActionResult {
            let (p, _) = src.choose_mut("pair", &mut state.pairs).pre()?;
            let val: String = src.any("val");
            api::update(&p.key, &val);
            p.val = val;
            Ok(())
        }

        pub fn delete(src: &mut Source, state: &mut State) -> ActionResult {
            let (p, p_ix) = src.choose_mut("pair", &mut state.pairs).pre()?;
            api::delete(&p.key);
            state.pairs.swap_remove(p_ix);
            Ok(())
        }
    }

    #[test]
    fn action_driver() {
        check(|src| {
            let mut state = actions::State::default();
            src.repeat_select(
                "action",
                &["create", "update", "delete"],
                |src, variant, _ix| match variant {
                    "create" => actions::CreateAction::run(src, &mut state),
                    "update" => actions::UpdateAction::run(src, &mut state),
                    "delete" => actions::DeleteAction::run(src, &mut state),
                    _ => unreachable!(),
                },
            );
        });
    }

    fn action_step<State>(
        action: fn(&mut Source, &mut State) -> ActionResult,
        src: &mut Source,
        state: &mut State,
    ) -> Effect {
        let r = action(src, state);
        match r {
            Ok(()) => Effect::Success,
            Err(ActionError::Precondition) => Effect::Noop,
            Err(ActionError::Other) => Effect::Change,
        }
    }

    #[test]
    fn driver() {
        check(|src| {
            let mut state = actions::State::default();
            src.repeat_select(
                "action",
                &["create", "update", "delete"],
                |src, variant, _ix| match variant {
                    "create" => action_step(actions::create, src, &mut state),
                    "update" => action_step(actions::update, src, &mut state),
                    "delete" => action_step(actions::delete, src, &mut state),
                    _ => unreachable!(),
                },
            );
        });
    }
}
