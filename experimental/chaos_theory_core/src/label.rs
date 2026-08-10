use alloc::sync::Arc;
use core::{
    borrow::Borrow,
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    ops::Deref,
};

const LABEL_SIZE: usize = 24;
const LABEL_SIZE_INLINE: usize = LABEL_SIZE - 2;

#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<Option<Label>>() == LABEL_SIZE);
const _: () = assert!(size_of::<Option<Label>>() == size_of::<alloc::string::String>());

#[derive(Clone, Eq)]
pub(crate) enum Label {
    Inline(([u8; LABEL_SIZE_INLINE], u8)),
    Heap(Arc<[u8]>),
}

impl From<&str> for Label {
    fn from(value: &str) -> Self {
        Self::from(value.as_bytes())
    }
}

impl From<&[u8]> for Label {
    fn from(value: &[u8]) -> Self {
        let n = value.len();
        if n <= LABEL_SIZE_INLINE {
            let mut buf = [0u8; LABEL_SIZE_INLINE];
            buf[..n].copy_from_slice(value);
            Self::Inline((buf, n as u8))
        } else {
            Self::Heap(value.into())
        }
    }
}

impl AsRef<[u8]> for Label {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Inline((buf, n)) => &buf[..(*n as usize)],
            Self::Heap(s) => s,
        }
    }
}

impl Default for Label {
    fn default() -> Self {
        Self::Inline(Default::default())
    }
}

impl Debug for Label {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(self.as_ref(), f)
    }
}

impl Display for Label {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for chunk in self.as_ref().utf8_chunks() {
            f.write_str(chunk.valid())?;
            if !chunk.invalid().is_empty() {
                f.write_str("\u{FFFD}")?;
            }
        }
        Ok(())
    }
}

impl Borrow<[u8]> for Label {
    fn borrow(&self) -> &[u8] {
        self.as_ref()
    }
}

impl Deref for Label {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl Ord for Label {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl PartialOrd for Label {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for Label {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state);
    }
}
