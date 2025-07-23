use chaos_theory::{Source, make};

pub fn prop_fuzz_target_1(src: &mut Source) {
    prop_fuzz_target_1_vec(src);
}

fn _prop_fuzz_target_1_string(src: &mut Source) {
    let s: String = src.any("s");
    assert_ne!(&s, "banana!");
}

#[expect(clippy::collapsible_if, clippy::collapsible_else_if)]
fn prop_fuzz_target_1_vec(src: &mut Source) {
    let s: Vec<u8> = src.any_of("s", make::vec_with_size(make::arbitrary(), 7..));
    // TODO: figure out why this one does not work:
    // let s: Vec<u8> = src.any("s");
    if false {
        assert_ne!(s, b"banana!");
    } else {
        if s.len() >= 7 {
            if s[0] == b'b' {
                if s[1] == b'a' {
                    if s[2] == b'n' {
                        if s[3] == b'a' {
                            if s[4] == b'n' {
                                if s[5] == b'a' {
                                    if s[6] == b'!' {
                                        panic!("success!");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn check_fuzz_target_1() {
    chaos_theory::check(prop_fuzz_target_1);
}

#[test]
#[ignore = "should not run unless explicitly invoked"]
fn write_seeds_fuzz_target_1() {
    for _ in 0..32 {
        chaos_theory::fuzz_write_seed("fuzz/corpus/fuzz_target_1/", prop_fuzz_target_1).unwrap();
    }
}
