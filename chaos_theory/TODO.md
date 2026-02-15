# TODO

## docs

- into
- usage examples
- FAQ
  - logging in tests, `--no-capture`, `println`
- guide on how to write a generator impl for struct and for enum
- `SKILL.md`

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
