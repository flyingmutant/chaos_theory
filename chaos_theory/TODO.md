# TODO

## docs

- crates.io + docs.rs badges
- better "main" example
- usage examples
- skill or one-pager for coding agents

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
- CString
- OSString
- PathBuf
- IpAddr/SocketAddr
- range types, Bound
- MaybeUninit
- permutation/shuffle/subsequence/random chunking
- recursive

## features

- derive macro
- consider saving failures, at least temporarily
