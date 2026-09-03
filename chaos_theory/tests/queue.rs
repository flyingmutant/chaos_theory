//! Test a deliberately buggy circular queue with `chaos_theory`.

extern crate alloc;
use alloc::collections::VecDeque;

use chaos_theory::{Effect, check, make};

/// Integer queue with a fixed maximum size.
struct Queue {
    buffer: Vec<i32>,
    input: usize,
    output: usize,
}

impl Queue {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0; capacity + 1],
            input: 0,
            output: 0,
        }
    }

    /// Precondition: queue is not empty.
    fn get(&mut self) -> i32 {
        let value = self.buffer[self.output];
        self.output = (self.output + 1) % self.buffer.len();
        value
    }

    /// Precondition: queue is not full.
    fn put(&mut self, value: i32) {
        self.buffer[self.input] = value;
        self.input = (self.input + 1) % self.buffer.len();
    }

    fn len(&self) -> usize {
        // Each of these is subtly wrong:
        (self.input - self.output) % self.buffer.len()
        // self.input.abs_diff(self.output)
        // self.input.wrapping_sub(self.output) % self.buffer.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[test]
#[should_panic]
fn queue_matches_model() {
    check(|src| {
        let capacity: usize = src.any_of("capacity", make::int_in(1..=1000));
        let mut queue = Queue::new(capacity);
        let mut model = VecDeque::new();

        src.repeat("step", |src| {
            src.select("action", &["get", "put"], |src, action, _| {
                match action {
                    "get" => {
                        if queue.is_empty() {
                            return Effect::Noop;
                        }

                        let value = queue.get();
                        let expected = model.pop_front().expect("model queue is empty");
                        assert_eq!(value, expected, "got an invalid value");
                        src.log_value("value", &value);
                    }
                    "put" => {
                        if queue.len() == capacity {
                            return Effect::Noop;
                        }

                        let value: i32 = src.any("value");
                        queue.put(value);
                        model.push_back(value);
                    }
                    _ => unreachable!(),
                }

                assert_eq!(queue.len(), model.len(), "queue size mismatch");
                Effect::Success
            })
        });
    });
}
