# TODO

## docs

- less "boundary values" , more "explore small" (focus on smart data generation)
- "repro"
- "safe, meaningful cuts"
- "simple idea" wording
- LICENSE link in README
- doc links in README
- include docs in `_docs` module (& ensure proper links)
- guide: repeat "step" not "steps"
- guide: don't encourage copy-pasting/agents
- generators: bad seed examples, reserve vec
- wrap lines
- remove "intentionally brief" notes
- fuzzing example uses vec<u8>
- `CHAOS_THEORY_REPLAY` make auto-filter tests? or just mention the test
- faq: proptest "centers"?
- reduction in code vs minimization in docs (do a pass and propose something good)
- `SKILL.md`
- update changelog

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
