# TODO

## docs

- overall quality pass
  - faq: proptest "centers"?
  - remove explicit types from examples where don't need them
  - reduction in code vs minimization in docs (do a pass and propose something good)
- update changelog
- examples for functions
- design notes, better/more comparisons

- skills

## API

- consider renaming `_with_size` to `_n`
- consider `make::string` + `make::string_of`
  - dot works better + type inference is simpler + python is simpler
  - what to do with `int*` and `float*`?
- expose generator types?

## generators

- external
  - uuid
  - serde_json
  - bytes
- u128
- CString/Cstr
- OSString/OSstr
- PathBuf/Path
- IpAddr/SocketAddr
- range types, Bound
- MaybeUninit
- permutation/shuffle/subsequence/random chunking
- recursive

## features

- derive macro
- consider saving failures, at least temporarily
