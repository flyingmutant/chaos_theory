// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use core::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6},
    num::NonZero,
};

use crate::{Arbitrary, Generator, SourceEx, make, range::Range};

const IPV4_SEEDS: &[Ipv4Addr] = &[
    Ipv4Addr::UNSPECIFIED,         // unspecified address
    Ipv4Addr::LOCALHOST,           // loopback address
    Ipv4Addr::BROADCAST,           // limited broadcast address
    Ipv4Addr::new(10, 0, 0, 1),    // private 10.0.0.0/8 range
    Ipv4Addr::new(172, 16, 0, 1),  // private 172.16.0.0/12 range
    Ipv4Addr::new(192, 168, 0, 1), // private 192.168.0.0/16 range
    Ipv4Addr::new(169, 254, 0, 1), // link-local range
    Ipv4Addr::new(192, 0, 2, 1),   // TEST-NET-1 documentation range
    Ipv4Addr::new(224, 0, 0, 1),   // all-hosts multicast group
];

const IPV6_SEEDS: &[Ipv6Addr] = &[
    Ipv6Addr::UNSPECIFIED,                               // unspecified address
    Ipv6Addr::LOCALHOST,                                 // loopback address
    Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1),          // link-local unicast range
    Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 1),          // unique-local unicast range
    Ipv6Addr::new(0xff02, 0, 0, 0, 0, 0, 0, 1),          // link-local all-nodes multicast group
    Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1),      // documentation range
    Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc000, 0x201), // IPv4-mapped 192.0.2.1
];

const PORT_SEEDS: &[u16] = &[
    20,    // FTP data
    21,    // FTP control
    22,    // SSH
    25,    // SMTP
    53,    // DNS
    67,    // DHCP server
    68,    // DHCP client
    80,    // HTTP
    110,   // POP3
    123,   // NTP
    143,   // IMAP
    161,   // SNMP
    389,   // LDAP
    443,   // HTTPS
    445,   // SMB
    587,   // SMTP message submission
    636,   // LDAPS
    993,   // IMAPS
    995,   // POP3S
    1433,  // Microsoft SQL Server
    1883,  // MQTT
    3306,  // MySQL
    3389,  // RDP
    5432,  // PostgreSQL
    5672,  // AMQP
    6379,  // Redis
    8080,  // HTTP alternative
    27017, // MongoDB
];

#[derive(Debug)]
struct UniformOctet;

impl Generator for UniformOctet {
    type Item = u8;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        src.as_mut().choose_value(
            Range::new_raw(0, u64::from(u8::MAX)),
            example.copied().map(u64::from),
            false,
        ) as u8
    }
}

#[derive(Debug)]
struct Ipv4Addr_<G> {
    octets: G,
}

impl<G: Generator<Item = [u8; 4]>> Generator for Ipv4Addr_<G> {
    type Item = Ipv4Addr;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example = example.map(Ipv4Addr::octets);
        let octets = self.octets.next(src, example.as_ref());
        Ipv4Addr::from(octets)
    }
}

impl Arbitrary for Ipv4Addr {
    fn arbitrary() -> impl Generator<Item = Self> {
        Ipv4Addr_ {
            octets: make::array(UniformOctet),
        }
        .seeded(IPV4_SEEDS, true)
    }
}

#[derive(Debug)]
struct Ipv6Addr_<G> {
    octets: G,
}

impl<G: Generator<Item = [u8; 16]>> Generator for Ipv6Addr_<G> {
    type Item = Ipv6Addr;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example = example.map(Ipv6Addr::octets);
        let octets = self.octets.next(src, example.as_ref());
        Ipv6Addr::from(octets)
    }
}

impl Arbitrary for Ipv6Addr {
    fn arbitrary() -> impl Generator<Item = Self> {
        Ipv6Addr_ {
            octets: make::array(UniformOctet),
        }
        .seeded(IPV6_SEEDS, true)
    }
}

#[derive(Debug)]
struct IpAddr_<G4, G6> {
    v4: G4,
    v6: G6,
}

impl<G4: Generator<Item = Ipv4Addr>, G6: Generator<Item = Ipv6Addr>> Generator for IpAddr_<G4, G6> {
    type Item = IpAddr;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_index = example.map(|addr| match addr {
            IpAddr::V4(_) => 0,
            IpAddr::V6(_) => 1,
        });
        let variants = ["V4", "V6"];
        let variants_num = NonZero::new(variants.len()).expect("internal error: no variants");
        src.select(
            "<ip-addr>",
            example_index,
            variants_num,
            |ix| variants[ix],
            |src, variant, _ix| match variant {
                "V4" => {
                    let example = match example {
                        Some(IpAddr::V4(addr)) => Some(addr),
                        _ => None,
                    };
                    IpAddr::V4(self.v4.next(src, example))
                }
                "V6" => {
                    let example = match example {
                        Some(IpAddr::V6(addr)) => Some(addr),
                        _ => None,
                    };
                    IpAddr::V6(self.v6.next(src, example))
                }
                _ => unreachable!(),
            },
        )
    }
}

impl Arbitrary for IpAddr {
    fn arbitrary() -> impl Generator<Item = Self> {
        IpAddr_ {
            v4: Ipv4Addr::arbitrary(),
            v6: Ipv6Addr::arbitrary(),
        }
    }
}

fn port() -> impl Generator<Item = u16> {
    u16::arbitrary().seeded(PORT_SEEDS, false)
}

#[derive(Debug)]
struct SocketAddrV4_<GI, GP> {
    ip: GI,
    port: GP,
}

impl<GI: Generator<Item = Ipv4Addr>, GP: Generator<Item = u16>> Generator
    for SocketAddrV4_<GI, GP>
{
    type Item = SocketAddrV4;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_port = example.map(SocketAddrV4::port);
        let ip = src.any_of("ip", &self.ip, example.map(SocketAddrV4::ip));
        let port = src.any_of("port", &self.port, example_port.as_ref());
        SocketAddrV4::new(ip, port)
    }
}

impl Arbitrary for SocketAddrV4 {
    fn arbitrary() -> impl Generator<Item = Self> {
        SocketAddrV4_ {
            ip: Ipv4Addr::arbitrary(),
            port: port(),
        }
    }
}

#[derive(Debug)]
struct SocketAddrV6_<GI, GP, GF, GS> {
    ip: GI,
    port: GP,
    flowinfo: GF,
    scope_id: GS,
}

impl<GI, GP, GF, GS> Generator for SocketAddrV6_<GI, GP, GF, GS>
where
    GI: Generator<Item = Ipv6Addr>,
    GP: Generator<Item = u16>,
    GF: Generator<Item = u32>,
    GS: Generator<Item = u32>,
{
    type Item = SocketAddrV6;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_port = example.map(SocketAddrV6::port);
        let example_flowinfo = example.map(SocketAddrV6::flowinfo);
        let example_scope_id = example.map(SocketAddrV6::scope_id);
        let ip = src.any_of("ip", &self.ip, example.map(SocketAddrV6::ip));
        let port = src.any_of("port", &self.port, example_port.as_ref());
        let flowinfo = src.any_of("flowinfo", &self.flowinfo, example_flowinfo.as_ref());
        let scope_id = src.any_of("scope_id", &self.scope_id, example_scope_id.as_ref());
        SocketAddrV6::new(ip, port, flowinfo, scope_id)
    }
}

impl Arbitrary for SocketAddrV6 {
    fn arbitrary() -> impl Generator<Item = Self> {
        SocketAddrV6_ {
            ip: Ipv6Addr::arbitrary(),
            port: port(),
            flowinfo: u32::arbitrary(),
            scope_id: u32::arbitrary(),
        }
    }
}

#[derive(Debug)]
struct SocketAddr_<G4, G6> {
    v4: G4,
    v6: G6,
}

impl<G4: Generator<Item = SocketAddrV4>, G6: Generator<Item = SocketAddrV6>> Generator
    for SocketAddr_<G4, G6>
{
    type Item = SocketAddr;

    fn next(&self, src: &mut SourceEx, example: Option<&Self::Item>) -> Self::Item {
        let example_index = example.map(|addr| match addr {
            SocketAddr::V4(_) => 0,
            SocketAddr::V6(_) => 1,
        });
        let variants = ["V4", "V6"];
        let variants_num = NonZero::new(variants.len()).expect("internal error: no variants");
        src.select(
            "<socket-addr>",
            example_index,
            variants_num,
            |ix| variants[ix],
            |src, variant, _ix| match variant {
                "V4" => {
                    let example = match example {
                        Some(SocketAddr::V4(addr)) => Some(addr),
                        _ => None,
                    };
                    SocketAddr::V4(self.v4.next(src, example))
                }
                "V6" => {
                    let example = match example {
                        Some(SocketAddr::V6(addr)) => Some(addr),
                        _ => None,
                    };
                    SocketAddr::V6(self.v6.next(src, example))
                }
                _ => unreachable!(),
            },
        )
    }
}

impl Arbitrary for SocketAddr {
    fn arbitrary() -> impl Generator<Item = Self> {
        SocketAddr_ {
            v4: SocketAddrV4::arbitrary(),
            v6: SocketAddrV6::arbitrary(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        check,
        tests::{print_debug_examples, prop_smoke},
    };

    #[test]
    fn net_smoke() {
        check(|src| {
            prop_smoke(src, "Ipv4Addr", Ipv4Addr::arbitrary());
            prop_smoke(src, "Ipv6Addr", Ipv6Addr::arbitrary());
            prop_smoke(src, "IpAddr", IpAddr::arbitrary());
            prop_smoke(src, "SocketAddrV4", SocketAddrV4::arbitrary());
            prop_smoke(src, "SocketAddrV6", SocketAddrV6::arbitrary());
            prop_smoke(src, "SocketAddr", SocketAddr::arbitrary());
        });
    }

    #[test]
    fn socket_addr_examples() {
        print_debug_examples(SocketAddr::arbitrary(), None, Ord::cmp);
    }
}
