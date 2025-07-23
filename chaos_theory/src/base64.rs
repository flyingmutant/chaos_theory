// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Based on public domain code from https://github.com/zhicheng/base64/blob/master/base64.c
// Standard alphabet, no padding, unoptimized implementation.

const ENCODE: [u8; 64] = [
    b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', //
    b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P', //
    b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', //
    b'Y', b'Z', b'a', b'b', b'c', b'd', b'e', b'f', //
    b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', //
    b'o', b'p', b'q', b'r', b's', b't', b'u', b'v', //
    b'w', b'x', b'y', b'z', b'0', b'1', b'2', b'3', //
    b'4', b'5', b'6', b'7', b'8', b'9', b'+', b'/', //
];

#[expect(clippy::inconsistent_digit_grouping)]
const DECODE: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, // nul, soh, stx, etx, eot, enq, ack, bel
    255, 255, 255, 255, 255, 255, 255, 255, //  bs,  ht,  nl,  vt,  np,  cr,  so,  si
    255, 255, 255, 255, 255, 255, 255, 255, // dle, dc1, dc2, dc3, dc4, nak, syn, etb
    255, 255, 255, 255, 255, 255, 255, 255, // can,  em, sub, esc,  fs,  gs,  rs,  us
    255, 255, 255, 255, 255, 255, 255, 255, //  sp, '!', '"', '#', '$', '%', '&', '''
    255, 255, 255, 62_, 255, 255, 255, 63_, // '(', ')', '*', '+', ',', '-', '.', '/'
    52_, 53_, 54_, 55_, 56_, 57_, 58_, 59_, // '0', '1', '2', '3', '4', '5', '6', '7'
    60_, 61_, 255, 255, 255, 255, 255, 255, // '8', '9', ':', ';', '<', '=', '>', '?'
    255, 0__, 1__, 2__, 3__, 4__, 5__, 6__, // '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G'
    7__, 8__, 9__, 10_, 11_, 12_, 13_, 14_, // 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'
    15_, 16_, 17_, 18_, 19_, 20_, 21_, 22_, // 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W'
    23_, 24_, 25_, 255, 255, 255, 255, 255, // 'X', 'Y', 'Z', '[', '\', ']', '^', '_'
    255, 26_, 27_, 28_, 29_, 30_, 31_, 32_, // '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g'
    33_, 34_, 35_, 36_, 37_, 38_, 39_, 40_, // 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'
    41_, 42_, 43_, 44_, 45_, 46_, 47_, 48_, // 'p', 'q', 'r', 's', 't', 'u', 'v', 'w'
    49_, 50_, 51_, 255, 255, 255, 255, 255, // 'x', 'y', 'z', '{', '|', '}', '~', del
];

pub(crate) fn encoded_len(n: usize) -> usize {
    n / 3 * 4 + (n % 3 * 8).div_ceil(6)
}

pub(crate) fn decoded_len(n: usize) -> usize {
    n / 4 * 3 + (n % 4 * 6) / 8
}

pub(crate) fn encode<'out>(
    data: &[u8],
    out: &'out mut [u8],
) -> Result<&'out mut [u8], &'static str> {
    if out.len() < encoded_len(data.len()) {
        return Err("buffer too short");
    }
    let mut s = 0;
    let mut l = 0;
    let mut j = 0;
    for &c in data {
        match s {
            0 => {
                s = 1;
                out[j] = ENCODE[usize::from(c >> 2)]; // put 6 high
                j += 1;
            }
            1 => {
                s = 2;
                out[j] = ENCODE[usize::from(((l & 0x3) << 4) | (c >> 4))]; // put 2 low + 4 high
                j += 1;
            }
            2 => {
                s = 0;
                out[j] = ENCODE[usize::from(((l & 0xf) << 2) | (c >> 6))]; // put 4 low + 2 high
                j += 1;
                out[j] = ENCODE[usize::from(c & 0x3f)]; // put 6 low
                j += 1;
            }
            _ => unreachable!(),
        }
        l = c;
    }
    match s {
        0 => {}
        1 => {
            out[j] = ENCODE[usize::from((l & 0x3) << 4)]; // put 2 low
            j += 1;
        }
        2 => {
            out[j] = ENCODE[usize::from((l & 0xf) << 2)]; // put 4 low
            j += 1;
        }
        _ => unreachable!(),
    }
    Ok(&mut out[j..])
}

pub(crate) fn decode<'out>(
    data: &[u8],
    out: &'out mut [u8],
) -> Result<&'out mut [u8], &'static str> {
    if out.len() < decoded_len(data.len()) {
        return Err("buffer too short");
    }
    let mut l = 0;
    let mut j = 0;
    for (i, &e) in data.iter().enumerate() {
        if usize::from(e) >= DECODE.len() {
            return Err("invalid base64 character");
        }
        let c = DECODE[usize::from(e)];
        if c == 255 {
            return Err("invalid base64 character");
        }
        match i % 4 {
            0 => {
                l = c << 2; // take 6 high
            }
            1 => {
                out[j] = l | (c >> 4); // take 2 high
                j += 1;
                l = (c & 0xf) << 4; // take 4 low
            }
            2 => {
                out[j] = l | (c >> 2); // take 4 high
                j += 1;
                l = (c & 0x3) << 6; // take 2 low
            }
            3 => {
                out[j] = l | c; // take 6 low
                j += 1;
            }
            _ => unreachable!(),
        }
    }
    Ok(&mut out[j..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check, vdbg};

    #[test]
    fn wikipedia_golden() {
        let data = "Many hands make light work.";
        let mut buf = vec![0; encoded_len(data.len())];
        encode(data.as_bytes(), &mut buf).unwrap();
        assert_eq!(&buf, "TWFueSBoYW5kcyBtYWtlIGxpZ2h0IHdvcmsu".as_bytes()); // cspell:disable-line
    }

    #[test]
    fn encoded_decoded_len_roundtrip() {
        check(|src| {
            let n: u16 = src.any("n");
            let e = encoded_len(usize::from(n));
            let d = decoded_len(e);
            assert_eq!(usize::from(n), d);
        });
    }

    #[test]
    fn encode_decode_roundtrip() {
        check(|src| {
            let data: Vec<u8> = src.any("data");
            let mut buf = vec![0; encoded_len(data.len())];
            let rem1 = encode(&data, &mut buf).unwrap();
            assert!(rem1.is_empty());
            vdbg!(src, &buf);
            let mut out = vec![0; decoded_len(buf.len())];
            let rem2 = decode(&buf, &mut out).unwrap();
            assert!(rem2.is_empty());
            assert_eq!(data, out);
        });
    }
}
