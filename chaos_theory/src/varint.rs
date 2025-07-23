// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// SQLite varints: https://sqlite.org/src4/doc/trunk/www/varint.wiki

pub(crate) const MAX_SIZE: usize = 9;

const BUFFER_TOO_SHORT: &str = "buffer too short";

pub(crate) fn decode(buf: &[u8]) -> Result<(u64, &[u8]), &'static str> {
    let n = buf.len();
    if n < 1 {
        return Err(BUFFER_TOO_SHORT);
    }
    let a0 = buf[0];
    match a0 {
        0..=240 => Ok((u64::from(a0), &buf[1..])),
        241..=248 => {
            if n < 2 {
                return Err(BUFFER_TOO_SHORT);
            }
            Ok((
                240 + 256 * (u64::from(a0) - 241) + u64::from(buf[1]),
                &buf[2..],
            ))
        }
        249 => {
            if n < 3 {
                return Err(BUFFER_TOO_SHORT);
            }
            Ok((
                2288 + 256 * u64::from(buf[1]) + u64::from(buf[2]),
                &buf[3..],
            ))
        }
        _ => {
            let m = 3 + usize::from(a0 - 250);
            if n < 1 + m {
                return Err(BUFFER_TOO_SHORT);
            }
            let mut bytes = [0u8; 8];
            bytes[8 - m..].copy_from_slice(&buf[1..=m]);
            Ok((u64::from_be_bytes(bytes), &buf[1 + m..]))
        }
    }
}

pub(crate) fn encode(v: u64, buf: &mut [u8]) -> Result<&mut [u8], &'static str> {
    let n = buf.len();
    if v <= 240 {
        if n < 1 {
            return Err(BUFFER_TOO_SHORT);
        }
        buf[0] = v as u8;
        return Ok(&mut buf[1..]);
    }
    if v <= 2287 {
        if n < 2 {
            return Err(BUFFER_TOO_SHORT);
        }
        buf[0] = ((v - 240) / 256 + 241) as u8;
        buf[1] = ((v - 240) % 256) as u8;
        return Ok(&mut buf[2..]);
    }
    if v <= 67823 {
        if n < 3 {
            return Err(BUFFER_TOO_SHORT);
        }
        buf[0] = 249;
        buf[1] = ((v - 2288) / 256) as u8;
        buf[2] = ((v - 2288) % 256) as u8;
        return Ok(&mut buf[3..]);
    }
    // There is certainly a way to do this much faster.
    let m: usize = if v <= 16777215 {
        3
    } else if v <= 4294967295 {
        4
    } else if v <= 1099511627775 {
        5
    } else if v <= 281474976710655 {
        6
    } else if v <= 72057594037927935 {
        7
    } else {
        8
    };
    if n < 1 + m {
        return Err(BUFFER_TOO_SHORT);
    }
    let bytes = v.to_be_bytes();
    buf[0] = 250 + (m - 3) as u8;
    buf[1..=m].copy_from_slice(&bytes[8 - m..]);
    Ok(&mut buf[m + 1..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, vdbg};

    #[test]
    fn encode_decode_roundtrip() {
        check(|src| {
            let mut buf = [0; MAX_SIZE];
            let v: u64 = src.any("v");
            let v_rem = encode(v, &mut buf).unwrap().len();
            vdbg!(src, buf);
            let (u, rem) = decode(&buf).unwrap();
            assert_eq!(v, u);
            assert_eq!(v_rem, rem.len());
        });
    }
}
