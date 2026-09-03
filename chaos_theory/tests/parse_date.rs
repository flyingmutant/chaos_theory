//! Test a deliberately buggy date parser with `chaos_theory`.

use chaos_theory::{check, make};

/// Parses dates in the `YYYY-MM-DD` format.
fn parse_date(date: &str) -> Result<(u16, u8, u8), &'static str> {
    if date.len() != 10 {
        return Err("date has the wrong length");
    }

    if date.get(4..5) != Some("-") || date.get(7..8) != Some("-") {
        return Err("date has invalid separators");
    }

    let year = date
        .get(0..4)
        .ok_or("invalid year")?
        .parse()
        .map_err(|_| "invalid year")?;
    let month = date
        .get(6..7)
        .ok_or("invalid month")?
        .parse()
        .map_err(|_| "invalid month")?;
    let day = date
        .get(8..10)
        .ok_or("invalid day")?
        .parse()
        .map_err(|_| "invalid day")?;

    Ok((year, month, day))
}

#[test]
#[should_panic]
fn parse_date_roundtrip() {
    check(|src| {
        let year: u16 = src.any_of("year", make::int_in(0..=9999));
        let month: u8 = src.any_of("month", make::int_in(1..=12));
        let day: u8 = src.any_of("day", make::int_in(1..=31));

        let date = format!("{year:04}-{month:02}-{day:02}");
        let parsed = parse_date(&date).expect("generated date should parse");

        assert_eq!(parsed, (year, month, day), "got back the wrong date");
    });
}
