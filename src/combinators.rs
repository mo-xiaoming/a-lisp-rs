#![allow(unused)]

use crate::parser::{Cursor, Input};

fn is_ascii_digit(c: &str) -> bool {
    assert_eq!(c.len(), 1);
    const ASCII_DIGITS: &str = "0123456789";
    ASCII_DIGITS.contains(c)
}

fn is_ascii_alpha(c: &str) -> bool {
    assert_eq!(c.len(), 1);
    const ASCII_ALPHABETS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    ASCII_ALPHABETS.contains(c)
}

fn is_ascii_alphanumeric(c: &str) -> bool {
    assert_eq!(c.len(), 1);
    is_ascii_alpha(c) || is_ascii_digit(c)
}

fn is_ascii_whitespace(c: &str) -> bool {
    c.len() == 1 && c.chars().next().unwrap().is_ascii_whitespace()
}

pub(crate) trait ParseError {}

#[derive(Debug, Clone)]
struct EarlyEof<'i> {
    input: &'i Input<'i>,
    expected: String,
}

impl<'i> ParseError for EarlyEof<'i> {}

impl<'i> std::fmt::Display for EarlyEof<'i> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: expected `{}`, but reached EOF",
            self.input.path(),
            self.expected
        )
    }
}

pub fn skip_spaces<'i>(input: &'i Input<'i>, cursor: Cursor) -> Cursor {
    input
        .take_while(cursor, is_ascii_whitespace)
        .unwrap_or(cursor)
}

#[derive(Debug, Clone)]
pub struct ParseCharError<'i> {
    pub(crate) input: &'i Input<'i>,
    pub(crate) expected: String,
    pub(crate) got: Cursor,
}

impl<'i> ParseError for ParseCharError<'i> {}

impl<'i> std::fmt::Display for ParseCharError<'i> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: expected `{}`, but got `{}`",
            self.input.path(),
            self.expected,
            self.input
                .get_str_ref(self.got, self.got.advance(1))
                .unwrap()
        )
    }
}

fn char<'i>(
    input: &'i Input<'i>,
    cursor: Cursor,
    c: &str,
) -> Result<(Cursor, &'i str), ParseCharError<'i>> {
    assert!(!input.is_eof(cursor), "char parser reaches EOF");

    let (new_cursor, x) = input.take_n(cursor, 1).unwrap();
    assert_eq!(new_cursor, cursor.advance(1));

    if x == c {
        Ok((new_cursor, x))
    } else {
        Err(ParseCharError {
            input,
            expected: c.to_owned(),
            got: cursor,
        })
    }
}

pub fn lparen<'i>(
    input: &'i Input<'i>,
    cursor: Cursor,
) -> Result<(Cursor, ()), ParseCharError<'i>> {
    char(input, cursor, "(").map(|(i, _)| (i, ()))
}

pub fn rparen<'i>(
    input: &'i Input<'i>,
    cursor: Cursor,
) -> Result<(Cursor, ()), ParseCharError<'i>> {
    char(input, cursor, ")").map(|(i, _)| (i, ()))
}

#[derive(Debug, Clone)]
pub enum ParseIntegerError<'i> {
    Unexpected {
        input: &'i Input<'i>,
        got: Cursor,
    },
    PosOverflow {
        input: &'i Input<'i>,
        got: (Cursor, Cursor),
    },
    NegOverflow {
        input: &'i Input<'i>,
        got: (Cursor, Cursor),
    },
    UnexpectedEof {
        input: &'i Input<'i>,
        got: Cursor,
    },
    LeadingZero,
}

pub fn integer<'i>(
    input: &'i Input<'i>,
    cursor: Cursor,
) -> Result<(Cursor, i64), ParseIntegerError> {
    assert!(!input.is_eof(cursor), "integer parser reaches EOF");

    let (has_sign, neg) = match input.take_n(cursor, 1).unwrap().1 {
        "+" => (true, false),
        "-" => (true, true),
        c if "0123456789".contains(c) => (false, false),
        _ => return Err(ParseIntegerError::Unexpected { input, got: cursor }),
    };

    let digit_begin = if has_sign { cursor.advance(1) } else { cursor };
    if input.is_eof(digit_begin) {
        return Err(ParseIntegerError::UnexpectedEof {
            input,
            got: digit_begin,
        });
    }

    let digit_end =
        input
            .take_while(digit_begin, is_ascii_digit)
            .ok_or(ParseIntegerError::Unexpected {
                input,
                got: digit_begin,
            })?;

    let digits = input.get_str_ref(cursor, digit_end).unwrap();

    if digits.len() != 1 && digits.starts_with('0') {
        return Err(ParseIntegerError::LeadingZero);
    }

    let i = digits.parse::<i64>().map_err(|e| match e.kind() {
        std::num::IntErrorKind::PosOverflow => ParseIntegerError::PosOverflow {
            input,
            got: (cursor, digit_end),
        },
        std::num::IntErrorKind::NegOverflow => ParseIntegerError::NegOverflow {
            input,
            got: (cursor, digit_end),
        },
        _ => unreachable!("unexpected {{integer}} parsing error"),
    })?;

    Ok((digit_end, i))
}

#[derive(Debug, Clone, Copy)]
pub struct ParseIdentifierError<'i> {
    pub(crate) input: &'i Input<'i>,
    pub(crate) got: Cursor,
}

impl<'i> ParseError for ParseIdentifierError<'i> {}

impl<'i> std::fmt::Display for ParseIdentifierError<'i> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: expected `{{identifier}}`, but got `{}`",
            self.input.path(),
            self.input
                .get_str_ref(self.got, self.got.advance(1))
                .unwrap()
        )
    }
}

pub fn identifier<'i>(
    input: &'i Input<'i>,
    cursor: Cursor,
) -> Result<(Cursor, &'i str), ParseIdentifierError> {
    assert!(!input.is_eof(cursor), "identifier parser reaches EOF");

    let new_cursor = input
        .take_while(cursor, |c| is_ascii_alphanumeric(c) || c == "_")
        .filter(|_| !is_ascii_digit(input.get_str_ref(cursor, cursor.advance(1)).unwrap()))
        .ok_or(ParseIdentifierError { input, got: cursor })?;

    Ok((new_cursor, input.get_str_ref(cursor, new_cursor).unwrap()))
}

#[derive(Debug, Clone, Copy)]
pub enum ParseSymbolError<'i> {
    NoLeadingHash {
        input: &'i Input<'i>,
        got: Cursor,
    },
    UnexpectedEof {
        input: &'i Input<'i>,
        got: Cursor,
    },
    Unknown {
        input: &'i Input<'i>,
        got: (Cursor, Cursor),
    },
}

#[derive(Debug, Clone, Copy, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Symbol {
    True,
    False,
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Symbol {
    fn as_str(&self) -> &'static str {
        match self {
            Symbol::True => "#t",
            Symbol::False => "#f",
        }
    }
}

pub fn symbol<'i>(
    input: &'i Input<'i>,
    cursor: Cursor,
) -> Result<(Cursor, Symbol), ParseSymbolError> {
    assert!(!input.is_eof(cursor), "symbol parser reaches EOF");

    if input.get_str_ref(cursor, cursor.advance(1)) != Some("#") {
        return Err(ParseSymbolError::NoLeadingHash { input, got: cursor });
    }

    let after_hash_cursor = cursor.advance(1);
    if input.is_eof(after_hash_cursor) {
        return Err(ParseSymbolError::UnexpectedEof {
            input,
            got: after_hash_cursor,
        });
    }
    let (end_cursor, s) =
        identifier(input, after_hash_cursor).map_err(|ParseIdentifierError { input, got }| {
            ParseSymbolError::UnexpectedEof { input, got }
        })?;
    match s {
        "t" => Ok((end_cursor, Symbol::True)),
        "f" => Ok((end_cursor, Symbol::False)),
        _ => Err(ParseSymbolError::Unknown {
            input,
            got: (cursor, end_cursor),
        }),
    }
}

#[cfg(test)]
mod test {
    use crate::parser::RawInput;

    use super::*;

    #[test]
    fn test_skip_spaces() {
        let sf = |s: &str, i: usize| {
            let input = RawInput::new(s.to_owned());
            let input = input.unicode_input();
            let cursor = skip_spaces(&input, input.begin());
            assert_eq!(cursor, Cursor::from(i));
        };

        for s in [("", 0), ("a", 0), (" ", 1), ("  b", 2), (" \n \n a", 5)] {
            sf(s.0, s.1);
        }
    }

    fn empty_raw_input() -> RawInput {
        RawInput::new(String::new())
    }

    fn non_empty_raw_input() -> RawInput {
        RawInput::new("hi".to_owned())
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF")]
    fn lparen_panic_on_empty_input() {
        let input = empty_raw_input();
        let input = input.unicode_input();
        let _r = lparen(&input, input.begin());
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF")]
    fn lparen_panic_on_eof() {
        let input = non_empty_raw_input();
        let input = input.unicode_input();
        let _r = lparen(&input, input.end());
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF")]
    fn rparen_panic_on_empty_input() {
        let input = empty_raw_input();
        let input = input.unicode_input();
        let _r = rparen(&input, input.begin());
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF")]
    fn rparen_panic_on_eof() {
        let input = non_empty_raw_input();
        let input = input.unicode_input();
        let _r = rparen(&input, input.end());
    }

    fn test_paren_ok<P>(parser: P, input: &RawInput, cursor: Cursor)
    where
        P: for<'i> Fn(&'i Input<'i>, Cursor) -> Result<(Cursor, ()), ParseCharError<'i>>,
    {
        let _input = input.unicode_input();
        let r = parser(&_input, cursor);
        assert!(matches!(
            r,
            Ok((
                new_cursor, ()
            )) if new_cursor == cursor.advance(1)
        ),);
    }

    fn test_paren_err<P>(parser: P, c: &str, input: &RawInput, cursor: Cursor)
    where
        P: for<'i> Fn(&'i Input<'i>, Cursor) -> Result<(Cursor, ()), ParseCharError<'i>>,
    {
        let input = input.unicode_input();
        let r = parser(&input, cursor);
        assert!(
            matches!(r, Err(ParseCharError{ input, expected, got  }) if expected == c && got ==  cursor)
        );
    }

    #[test]
    fn test_parens() {
        let input = RawInput::new("(".to_owned());
        test_paren_ok(lparen, &input, Cursor::from(0));
        let input = RawInput::new("x(".to_owned());
        test_paren_ok(lparen, &input, Cursor::from(1));
        let input = RawInput::new("(xx".to_owned());
        test_paren_ok(lparen, &input, Cursor::from(0));
        let input = RawInput::new("(xx".to_owned());
        test_paren_err(lparen, "(", &input, Cursor::from(1));

        let input = RawInput::new(")".to_owned());
        test_paren_ok(rparen, &input, Cursor::from(0));
        let input = RawInput::new("x)".to_owned());
        test_paren_ok(rparen, &input, Cursor::from(1));
        let input = RawInput::new(")xx".to_owned());
        test_paren_ok(rparen, &input, Cursor::from(0));
        let input = RawInput::new(")xx".to_owned());
        test_paren_err(rparen, ")", &input, Cursor::from(1));
    }

    #[test]
    #[should_panic(expected = "integer parser reaches EOF")]
    fn integer_panics_on_empty_input() {
        let input = empty_raw_input();
        let input = input.unicode_input();
        let _r = integer(&input, input.begin());
    }

    #[test]
    #[should_panic(expected = "integer parser reaches EOF")]
    fn integer_panics_on_eof() {
        let input = non_empty_raw_input();
        let input = input.unicode_input();
        let _r = integer(&input, input.end());
    }

    #[test]
    fn test_normal_integers() {
        let sf = |(s, expected, advance, cursor): (&str, i64, usize, Cursor)| {
            let input = RawInput::new(s.to_owned());
            let input = input.unicode_input();
            let r = integer(&input, cursor);
            assert!(matches!(
                r,
                Ok((
                        c,
                        v,
                )) if c == cursor.advance(advance) && v == expected
            ));
        };

        for s in [
            ("0", 0, 1, Cursor::from(0)),
            ("+0", 0, 2, Cursor::from(0)),
            ("-0", 0, 2, Cursor::from(0)),
            ("12345", 12345, 5, Cursor::from(0)),
            ("12345abc", 12345, 5, Cursor::from(0)),
            ("+12345", 12345, 6, Cursor::from(0)),
            ("+12345abc", 12345, 6, Cursor::from(0)),
            ("-12345", -12345, 6, Cursor::from(0)),
            ("-12345abc", -12345, 6, Cursor::from(0)),
            ("12345", 345, 3, Cursor::from(2)),
            ("12345abc", 345, 3, Cursor::from(2)),
            ("12+345", 345, 4, Cursor::from(2)),
            ("12+345abc", 345, 4, Cursor::from(2)),
            ("12-345", -345, 4, Cursor::from(2)),
            ("12-345abc", -345, 4, Cursor::from(2)),
            ("120", 0, 1, Cursor::from(2)),
            ("12+0", 0, 2, Cursor::from(2)),
            ("12-0", 0, 2, Cursor::from(2)),
            (
                "9223372036854775807", // MAX
                i64::MAX,
                i64::MAX.to_string().len(),
                Cursor::from(0),
            ),
            (
                "+9223372036854775807", // MAX
                i64::MAX,
                i64::MAX.to_string().len() + 1,
                Cursor::from(0),
            ),
            (
                "-9223372036854775808", // MIN
                i64::MIN,
                i64::MIN.to_string().len(),
                Cursor::from(0),
            ),
        ] {
            sf(s);
        }
    }

    #[test]
    fn integer_starts_with_zero_is_not_allowed() {
        let input = RawInput::new("012345".to_owned());
        let input = input.unicode_input();
        let r = integer(&input, input.begin());
        assert!(matches!(r, Err(ParseIntegerError::LeadingZero)));
    }

    #[test]
    fn integer_has_no_digits_after_sign_is_not_allowed() {
        for s in ["+", "-"] {
            let input = RawInput::new(s.to_owned());
            let input = input.unicode_input();
            let r = integer(&input, input.begin());
            assert!(matches!(
                r,
                Err(ParseIntegerError::UnexpectedEof {
                    got,
                    ..
                }) if got == input.begin().advance(1)
            ));
        }
    }

    #[test]
    fn integer_has_no_digits_is_not_allowed() {
        let input = RawInput::new("abc".to_owned());
        let input = input.unicode_input();
        let r = integer(&input, input.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::Unexpected {
                got,
                ..
            }) if got == input.begin()));
    }

    #[test]
    fn integer_overflows() {
        let input = RawInput::new("9223372036854775808".to_owned()); // MAX + 1
        let input = input.unicode_input();
        let r = integer(&input, input.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::PosOverflow {
                got, ..
            }) if got == (input.begin(), input.end())
        ));
    }

    #[test]
    fn integer_overflows_consume_all_digits() {
        let input = RawInput::new("92233720368547758081".to_owned()); // MAX + N
        let input = input.unicode_input();
        let r = integer(&input, input.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::PosOverflow {
                got, ..
            }) if got == (input.begin(), input.end())
        ));
    }

    #[test]
    fn integer_underflow() {
        let input = RawInput::new("-9223372036854775809".to_owned()); // MIN - 1
        let input = input.unicode_input();
        let r = integer(&input, input.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::NegOverflow {
                got, ..
            }) if got == (input.begin(), input.end())
        ));
    }

    #[test]
    fn integer_underflow_consume_all_digits() {
        let input = RawInput::new("-92233720368547758091".to_owned()); // MIN - N
        let input = input.unicode_input();
        let r = integer(&input, input.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::NegOverflow {
                got, ..
            }) if got == (input.begin(), input.end())
        ));
    }

    #[test]
    #[should_panic(expected = "identifier parser reaches EOF")]
    fn identifier_panics_on_empty_input() {
        let input = empty_raw_input();
        let input = input.unicode_input();
        let _r = identifier(&input, input.begin());
    }

    #[test]
    #[should_panic(expected = "identifier parser reaches EOF")]
    fn identifier_panics_on_eof() {
        let input = non_empty_raw_input();
        let input = input.unicode_input();
        let _r = identifier(&input, input.end());
    }

    #[test]
    fn test_identifier() {
        let sf = |s: &str| {
            let input = RawInput::new(s.to_owned());
            let input = input.unicode_input();
            let r = identifier(&input, input.begin());
            assert!(matches!(
                r,
                Ok((
                    c,
                    v
                )) if c == input.begin().advance(s.len()) && v == s
            ));
        };

        for s in ["abc", "_ab", "_3a", "a_b_c", "a_3", "a3_", "__", "_", "__a"] {
            sf(s);
        }

        let ef = |s: &str| {
            let input = RawInput::new(s.to_owned());
            let input = input.unicode_input();
            let r = identifier(&input, input.begin());
            assert!(matches!(
                r,
                Err(ParseIdentifierError {
                    got,..
                }) if got == input.begin()
            ));
        };

        for s in ["3ab", "3_ab", "#ab", "@ab"] {
            ef(s);
        }
    }

    #[test]
    #[should_panic(expected = "symbol parser reaches EOF")]
    fn symbol_panics_on_empty_input() {
        let input = empty_raw_input();
        let input = input.unicode_input();
        let _r = symbol(&input, input.begin());
    }

    #[test]
    #[should_panic(expected = "symbol parser reaches EOF")]
    fn symbol_panics_on_eof() {
        let input = non_empty_raw_input();
        let input = input.unicode_input();
        let _r = symbol(&input, input.end());
    }

    #[test]
    fn test_symbol() {
        let sf = |s: &str, sym: Symbol| {
            let input = RawInput::new(s.to_owned());
            let input = input.unicode_input();
            let r = symbol(&input, input.begin());
            assert!(matches!(
                r,
                Ok((
                    c,
                    v
                )) if c == input.begin().advance(sym.as_str().len()) && v == sym
            ),);
        };

        for (s, m) in [
            ("#t", Symbol::True),
            ("#t(", Symbol::True),
            ("#f", Symbol::False),
            ("#f(", Symbol::False),
        ] {
            sf(s, m);
        }
    }

    #[test]
    fn test_symbol_hash_only() {
        let sf = |s: &str| {
            let input = RawInput::new(s.to_owned());
            let input = input.unicode_input();
            let r = symbol(&input, input.begin());
            assert!(matches!(
                r,
                Err(ParseSymbolError::UnexpectedEof{
                   got,
                   ..
                }) if got == input.begin().advance(1)
            ),);
        };

        for s in ["#", "#3", "##"] {
            sf(s);
        }
    }

    #[test]
    fn test_symbol_no_hash() {
        let input = RawInput::new("a#".to_owned());
        let input = input.unicode_input();
        let r = symbol(&input, input.begin());
        assert!(matches!(
            r,
            Err(ParseSymbolError::NoLeadingHash{
               got,
               ..
            }) if got == input.begin()
        ),);
    }

    #[test]
    fn test_symbol_unknown() {
        let input = RawInput::new("#xyz".to_owned());
        let input = input.unicode_input();
        let r = symbol(&input, input.begin());
        assert!(matches!(
            r,
            Err(ParseSymbolError::Unknown{
               got,
               ..
            }) if got == (input.begin(), input.begin().advance(4))
        ),);
    }
}
