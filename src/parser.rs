#![allow(unused)]

use std::borrow::Cow;
use unicode_segmentation::UnicodeSegmentation;

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

trait ParseError {}

#[derive(Debug, Clone)]
struct EarlyEof<'i> {
    source_file: &'i SourceFile<'i>,
    expected: String,
}

impl<'i> ParseError for EarlyEof<'i> {}

impl<'i> std::fmt::Display for EarlyEof<'i> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: expected `{}`, but reached EOF",
            self.source_file.path(),
            self.expected
        )
    }
}

pub fn skip_spaces<'i>(source_file: &'i SourceFile<'i>, cursor: Cursor) -> Cursor {
    source_file
        .take_while(cursor, is_ascii_whitespace)
        .unwrap_or(cursor)
}

#[derive(Debug, Clone)]
pub struct ParseCharError<'i> {
    source_file: &'i SourceFile<'i>,
    expected: String,
    got: Cursor,
}

impl<'i> ParseError for ParseCharError<'i> {}

impl<'i> std::fmt::Display for ParseCharError<'i> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: expected `{}`, but got `{}`",
            self.source_file.path(),
            self.expected,
            self.source_file
                .get_str_ref(self.got, self.got.advance(1))
                .unwrap()
        )
    }
}

fn char<'i>(
    source_file: &'i SourceFile<'i>,
    cursor: Cursor,
    c: &str,
) -> Result<(Cursor, &'i str), ParseCharError<'i>> {
    assert!(!source_file.is_eof(cursor), "char parser reaches EOF");

    let (new_cursor, x) = source_file.take_n(cursor, 1).unwrap();
    assert_eq!(new_cursor, cursor.advance(1));

    if x == c {
        Ok((new_cursor, x))
    } else {
        Err(ParseCharError {
            source_file,
            expected: c.to_owned(),
            got: cursor,
        })
    }
}

pub fn lparen<'i>(
    source_file: &'i SourceFile<'i>,
    cursor: Cursor,
) -> Result<(Cursor, ()), ParseCharError<'i>> {
    char(source_file, cursor, "(").map(|(i, _)| (i, ()))
}

pub fn rparen<'i>(
    source_file: &'i SourceFile<'i>,
    cursor: Cursor,
) -> Result<(Cursor, ()), ParseCharError<'i>> {
    char(source_file, cursor, ")").map(|(i, _)| (i, ()))
}

#[derive(Debug, Clone)]
pub enum ParseIntegerError<'i> {
    Unexpected {
        source_file: &'i SourceFile<'i>,
        got: Cursor,
    },
    PosOverflow {
        source_file: &'i SourceFile<'i>,
        got: (Cursor, Cursor),
    },
    NegOverflow {
        source_file: &'i SourceFile<'i>,
        got: (Cursor, Cursor),
    },
    UnexpectedEof {
        source_file: &'i SourceFile<'i>,
        got: Cursor,
    },
    LeadingZero,
}

pub fn integer<'i>(
    source_file: &'i SourceFile<'i>,
    cursor: Cursor,
) -> Result<(Cursor, i64), ParseIntegerError> {
    assert!(!source_file.is_eof(cursor), "integer parser reaches EOF");

    let (has_sign, neg) = match source_file.take_n(cursor, 1).unwrap().1 {
        "+" => (true, false),
        "-" => (true, true),
        c if "0123456789".contains(c) => (false, false),
        _ => {
            return Err(ParseIntegerError::Unexpected {
                source_file,
                got: cursor,
            })
        }
    };

    let digit_begin = if has_sign { cursor.advance(1) } else { cursor };
    if source_file.is_eof(digit_begin) {
        return Err(ParseIntegerError::UnexpectedEof {
            source_file,
            got: digit_begin,
        });
    }

    let digit_end = source_file.take_while(digit_begin, is_ascii_digit).ok_or(
        ParseIntegerError::Unexpected {
            source_file,
            got: digit_begin,
        },
    )?;

    let digits = source_file.get_str_ref(cursor, digit_end).unwrap();

    if digits.len() != 1 && digits.starts_with('0') {
        return Err(ParseIntegerError::LeadingZero);
    }

    let i = digits.parse::<i64>().map_err(|e| match e.kind() {
        std::num::IntErrorKind::PosOverflow => ParseIntegerError::PosOverflow {
            source_file,
            got: (cursor, digit_end),
        },
        std::num::IntErrorKind::NegOverflow => ParseIntegerError::NegOverflow {
            source_file,
            got: (cursor, digit_end),
        },
        _ => unreachable!("unexpected {{integer}} parsing error"),
    })?;

    Ok((digit_end, i))
}

#[derive(Debug, Clone, Copy)]
pub struct ParseIdentifierError<'i> {
    source_file: &'i SourceFile<'i>,
    got: Cursor,
}

impl<'i> ParseError for ParseIdentifierError<'i> {}

impl<'i> std::fmt::Display for ParseIdentifierError<'i> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: expected `{{identifier}}`, but got `{}`",
            self.source_file.path(),
            self.source_file
                .get_str_ref(self.got, self.got.advance(1))
                .unwrap()
        )
    }
}

pub fn identifier<'i>(
    source_file: &'i SourceFile<'i>,
    cursor: Cursor,
) -> Result<(Cursor, &'i str), ParseIdentifierError> {
    assert!(!source_file.is_eof(cursor), "identifier parser reaches EOF");

    let new_cursor = source_file
        .take_while(cursor, |c| is_ascii_alphanumeric(c) || c == "_")
        .filter(|_| !is_ascii_digit(source_file.get_str_ref(cursor, cursor.advance(1)).unwrap()))
        .ok_or(ParseIdentifierError {
            source_file,
            got: cursor,
        })?;

    Ok((
        new_cursor,
        source_file.get_str_ref(cursor, new_cursor).unwrap(),
    ))
}

#[derive(Debug, Clone, Copy)]
pub enum ParseSymbolError<'i> {
    NoLeadingHash {
        source_file: &'i SourceFile<'i>,
        got: Cursor,
    },
    UnexpectedEof {
        source_file: &'i SourceFile<'i>,
        got: Cursor,
    },
    Unknown {
        source_file: &'i SourceFile<'i>,
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
    source_file: &'i SourceFile<'i>,
    cursor: Cursor,
) -> Result<(Cursor, Symbol), ParseSymbolError> {
    assert!(!source_file.is_eof(cursor), "symbol parser reaches EOF");

    if source_file.get_str_ref(cursor, cursor.advance(1)) != Some("#") {
        return Err(ParseSymbolError::NoLeadingHash {
            source_file,
            got: cursor,
        });
    }

    let after_hash_cursor = cursor.advance(1);
    if source_file.is_eof(after_hash_cursor) {
        return Err(ParseSymbolError::UnexpectedEof {
            source_file,
            got: after_hash_cursor,
        });
    }
    let (end_cursor, s) = identifier(source_file, after_hash_cursor).map_err(
        |ParseIdentifierError { source_file, got }| ParseSymbolError::UnexpectedEof {
            source_file,
            got,
        },
    )?;
    match s {
        "t" => Ok((end_cursor, Symbol::True)),
        "f" => Ok((end_cursor, Symbol::False)),
        _ => Err(ParseSymbolError::Unknown {
            source_file,
            got: (cursor, end_cursor),
        }),
    }
}

#[cfg(test)]
mod lexemes_test {
    use super::*;

    #[test]
    fn test_skip_spaces() {
        let sf = |s: &str, i: usize| {
            let source_file = SourceFile::new(s.to_owned());
            let cursor = skip_spaces(&source_file, source_file.begin());
            assert_eq!(cursor, Cursor::from(i));
        };

        for s in [("", 0), ("a", 0), (" ", 1), ("  b", 2), (" \n \n a", 5)] {
            sf(s.0, s.1);
        }
    }

    fn empty_raw_source_file<'i>() -> SourceFile<'i> {
        SourceFile::new(String::new())
    }

    fn non_empty_raw_source_file<'i>() -> SourceFile<'i> {
        SourceFile::new("hi".to_owned())
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF")]
    fn lparen_panic_on_empty_source_file() {
        let source_file = empty_raw_source_file();
        let _r = lparen(&source_file, source_file.begin());
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF")]
    fn lparen_panic_on_eof() {
        let source_file = non_empty_raw_source_file();
        let _r = lparen(&source_file, source_file.end());
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF")]
    fn rparen_panic_on_empty_source_file() {
        let source_file = empty_raw_source_file();
        let _r = rparen(&source_file, source_file.begin());
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF")]
    fn rparen_panic_on_eof() {
        let source_file = non_empty_raw_source_file();
        let _r = rparen(&source_file, source_file.end());
    }

    fn test_paren_ok<P>(parser: P, source_file: &SourceFile, cursor: Cursor)
    where
        P: for<'i> Fn(&'i SourceFile<'i>, Cursor) -> Result<(Cursor, ()), ParseCharError<'i>>,
    {
        let r = parser(source_file, cursor);
        assert!(matches!(
            r,
            Ok((
                new_cursor, ()
            )) if new_cursor == cursor.advance(1)
        ),);
    }

    fn test_paren_err<P>(parser: P, c: &str, source_file: &SourceFile, cursor: Cursor)
    where
        P: for<'i> Fn(&'i SourceFile<'i>, Cursor) -> Result<(Cursor, ()), ParseCharError<'i>>,
    {
        let r = parser(source_file, cursor);
        assert!(
            matches!(r, Err(ParseCharError{ source_file, expected, got  }) if expected == c && got ==  cursor)
        );
    }

    #[test]
    fn test_parens() {
        let source_file = SourceFile::new("(".to_owned());
        test_paren_ok(lparen, &source_file, Cursor::from(0));
        let source_file = SourceFile::new("x(".to_owned());
        test_paren_ok(lparen, &source_file, Cursor::from(1));
        let source_file = SourceFile::new("(xx".to_owned());
        test_paren_ok(lparen, &source_file, Cursor::from(0));
        let source_file = SourceFile::new("(xx".to_owned());
        test_paren_err(lparen, "(", &source_file, Cursor::from(1));

        let source_file = SourceFile::new(")".to_owned());
        test_paren_ok(rparen, &source_file, Cursor::from(0));
        let source_file = SourceFile::new("x)".to_owned());
        test_paren_ok(rparen, &source_file, Cursor::from(1));
        let source_file = SourceFile::new(")xx".to_owned());
        test_paren_ok(rparen, &source_file, Cursor::from(0));
        let source_file = SourceFile::new(")xx".to_owned());
        test_paren_err(rparen, ")", &source_file, Cursor::from(1));
    }

    #[test]
    #[should_panic(expected = "integer parser reaches EOF")]
    fn integer_panics_on_empty_source_file() {
        let source_file = empty_raw_source_file();
        let _r = integer(&source_file, source_file.begin());
    }

    #[test]
    #[should_panic(expected = "integer parser reaches EOF")]
    fn integer_panics_on_eof() {
        let source_file = non_empty_raw_source_file();
        let _r = integer(&source_file, source_file.end());
    }

    #[test]
    fn test_normal_integers() {
        let sf = |(s, expected, advance, cursor): (&str, i64, usize, Cursor)| {
            let source_file = SourceFile::new(s.to_owned());
            let r = integer(&source_file, cursor);
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
        let source_file = SourceFile::new("012345".to_owned());
        let r = integer(&source_file, source_file.begin());
        assert!(matches!(r, Err(ParseIntegerError::LeadingZero)));
    }

    #[test]
    fn integer_has_no_digits_after_sign_is_not_allowed() {
        for s in ["+", "-"] {
            let source_file = SourceFile::new(s.to_owned());
            let r = integer(&source_file, source_file.begin());
            assert!(matches!(
                r,
                Err(ParseIntegerError::UnexpectedEof {
                    got,
                    ..
                }) if got == source_file.begin().advance(1)
            ));
        }
    }

    #[test]
    fn integer_has_no_digits_is_not_allowed() {
        let source_file = SourceFile::new("abc".to_owned());
        let r = integer(&source_file, source_file.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::Unexpected {
                got,
                ..
            }) if got == source_file.begin()));
    }

    #[test]
    fn integer_overflows() {
        let source_file = SourceFile::new("9223372036854775808".to_owned()); // MAX + 1
        let r = integer(&source_file, source_file.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::PosOverflow {
                got, ..
            }) if got == (source_file.begin(), source_file.end())
        ));
    }

    #[test]
    fn integer_overflows_consume_all_digits() {
        let source_file = SourceFile::new("92233720368547758081".to_owned()); // MAX + N
        let r = integer(&source_file, source_file.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::PosOverflow {
                got, ..
            }) if got == (source_file.begin(), source_file.end())
        ));
    }

    #[test]
    fn integer_underflow() {
        let source_file = SourceFile::new("-9223372036854775809".to_owned()); // MIN - 1
        let r = integer(&source_file, source_file.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::NegOverflow {
                got, ..
            }) if got == (source_file.begin(), source_file.end())
        ));
    }

    #[test]
    fn integer_underflow_consume_all_digits() {
        let source_file = SourceFile::new("-92233720368547758091".to_owned()); // MIN - N
        let r = integer(&source_file, source_file.begin());
        assert!(matches!(
            r,
            Err(ParseIntegerError::NegOverflow {
                got, ..
            }) if got == (source_file.begin(), source_file.end())
        ));
    }

    #[test]
    #[should_panic(expected = "identifier parser reaches EOF")]
    fn identifier_panics_on_empty_source_file() {
        let source_file = empty_raw_source_file();
        let _r = identifier(&source_file, source_file.begin());
    }

    #[test]
    #[should_panic(expected = "identifier parser reaches EOF")]
    fn identifier_panics_on_eof() {
        let source_file = non_empty_raw_source_file();
        let _r = identifier(&source_file, source_file.end());
    }

    #[test]
    fn test_identifier() {
        let sf = |s: &str| {
            let source_file = SourceFile::new(s.to_owned());
            let r = identifier(&source_file, source_file.begin());
            assert!(matches!(
                r,
                Ok((
                    c,
                    v
                )) if c == source_file.begin().advance(s.len()) && v == s
            ));
        };

        for s in ["abc", "_ab", "_3a", "a_b_c", "a_3", "a3_", "__", "_", "__a"] {
            sf(s);
        }

        let ef = |s: &str| {
            let source_file = SourceFile::new(s.to_owned());
            let r = identifier(&source_file, source_file.begin());
            assert!(matches!(
                r,
                Err(ParseIdentifierError {
                    got,..
                }) if got == source_file.begin()
            ));
        };

        for s in ["3ab", "3_ab", "#ab", "@ab"] {
            ef(s);
        }
    }

    #[test]
    #[should_panic(expected = "symbol parser reaches EOF")]
    fn symbol_panics_on_empty_source_file() {
        let source_file = empty_raw_source_file();
        let _r = symbol(&source_file, source_file.begin());
    }

    #[test]
    #[should_panic(expected = "symbol parser reaches EOF")]
    fn symbol_panics_on_eof() {
        let source_file = non_empty_raw_source_file();
        let _r = symbol(&source_file, source_file.end());
    }

    #[test]
    fn test_symbol() {
        let sf = |s: &str, sym: Symbol| {
            let source_file = SourceFile::new(s.to_owned());
            let r = symbol(&source_file, source_file.begin());
            assert!(matches!(
                r,
                Ok((
                    c,
                    v
                )) if c == source_file.begin().advance(sym.as_str().len()) && v == sym
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
            let source_file = SourceFile::new(s.to_owned());
            let r = symbol(&source_file, source_file.begin());
            assert!(matches!(
                r,
                Err(ParseSymbolError::UnexpectedEof{
                   got,
                   ..
                }) if got == source_file.begin().advance(1)
            ),);
        };

        for s in ["#", "#3", "##"] {
            sf(s);
        }
    }

    #[test]
    fn test_symbol_no_hash() {
        let source_file = SourceFile::new("a#".to_owned());
        let r = symbol(&source_file, source_file.begin());
        assert!(matches!(
            r,
            Err(ParseSymbolError::NoLeadingHash{
               got,
               ..
            }) if got == source_file.begin()
        ),);
    }

    #[test]
    fn test_symbol_unknown() {
        let source_file = SourceFile::new("#xyz".to_owned());
        let r = symbol(&source_file, source_file.begin());
        assert!(matches!(
            r,
            Err(ParseSymbolError::Unknown{
               got,
               ..
            }) if got == (source_file.begin(), source_file.begin().advance(4))
        ),);
    }
}
#[derive(Debug, Clone)]
pub struct SourceFile<'i> {
    path: std::path::PathBuf,
    raw: String,
    ucs: Vec<&'i str>,
}

#[derive(Debug, Default, Clone, Copy, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cursor(usize);

impl Cursor {
    pub const fn from(n: usize) -> Self {
        Self(n)
    }
    pub const fn advance(&self, n: usize) -> Self {
        Self(self.0 + n)
    }

    pub const fn get(&self) -> usize {
        self.0
    }
}

impl<'i> SourceFile<'i> {
    #[cfg(test)]
    pub fn new(raw: String) -> Self {
        let usc = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(raw.as_ptr(), raw.len()))
        }
        .graphemes(true)
        .collect();

        Self {
            path: std::path::PathBuf::from("mock"),
            raw,
            ucs: usc,
        }
    }

    pub fn path(&self) -> Cow<str> {
        self.path.as_os_str().to_string_lossy()
    }

    fn begin(&self) -> Cursor {
        Cursor::from(0)
    }

    fn end(&self) -> Cursor {
        Cursor::from(self.ucs.len())
    }

    fn get_str_ref(&self, begin: Cursor, end: Cursor) -> Option<&str> {
        if begin >= end || end > self.end() {
            None
        } else {
            Some(&self.raw[self.distance_from_begin(begin)..self.distance_from_begin(end)])
        }
    }

    fn distance_from_begin(&self, cursor: Cursor) -> usize {
        self.raw[cursor.get()..].as_ptr() as usize - self.raw.as_ptr() as usize
    }

    pub fn is_eof(&self, cursor: Cursor) -> bool {
        cursor >= self.end()
    }

    pub fn rest_len(&self, cursor: Cursor) -> usize {
        if self.is_eof(cursor) {
            0
        } else {
            self.ucs.len() - cursor.get()
        }
    }

    pub fn take_while<P>(&self, cursor: Cursor, p: P) -> Option<Cursor>
    where
        P: Fn(&str) -> bool,
    {
        if self.is_eof(cursor) {
            return None;
        }

        match self.ucs[cursor.get()..].iter().position(|&s| !p(s)) {
            Some(0) => None,
            Some(i) => Some(cursor.advance(i)),
            None => Some(self.end()),
        }
    }

    pub fn take_n(&self, cursor: Cursor, n: usize) -> Option<(Cursor, &str)> {
        let new_cursor = cursor.advance(n);
        self.get_str_ref(cursor, new_cursor)
            .map(|s| (new_cursor, s))
    }
}

#[cfg(test)]
mod source_file_test {
    use super::*;

    #[test]
    fn test_empty_source_file() {
        // empty source
        let empty = SourceFile::new(String::new());

        // 0th cursor
        let cursor = empty.begin();
        assert_eq!(cursor, empty.end());

        assert!(empty.is_eof(cursor));
        assert_eq!(empty.rest_len(cursor), 0);
        assert_eq!(empty.take_while(cursor, |c| c == " "), None);
        assert_eq!(empty.take_while(cursor, |c| c.is_empty()), None);
        assert_eq!(empty.take_n(cursor, 0), None);
        assert_eq!(empty.take_n(cursor, 1), None);

        // out of range cursor
        let cursor = cursor.advance(1);
        assert!(empty.is_eof(cursor));
        assert_eq!(empty.rest_len(cursor), 0);
        assert_eq!(empty.take_while(cursor, |c| c == " "), None);
        assert_eq!(empty.take_while(cursor, |c| c.is_empty()), None);
        assert_eq!(empty.take_n(cursor, 0), None);
        assert_eq!(empty.take_n(cursor, 1), None);
    }

    #[test]
    fn test_non_empty_source_file() {
        // non-empty source
        let source = SourceFile::new("aabc".to_owned());

        // 0th cursor
        let cursor = source.begin();
        assert_ne!(cursor, source.end());

        assert!(!source.is_eof(cursor));
        assert_eq!(source.rest_len(cursor), 4);
        assert_eq!(
            source.get_str_ref(cursor, source.take_while(cursor, |c| c == "a").unwrap()),
            Some("aa")
        );
        assert_eq!(source.take_while(cursor, |c| c == "b"), None);
        assert_eq!(source.take_n(cursor, 0), None);
        assert_eq!(source.take_n(cursor, 1), Some((cursor.advance(1), "a")));
        assert_eq!(source.take_n(cursor, 4), Some((cursor.advance(4), "aabc")));
        assert_eq!(source.take_n(cursor, 5), None);

        // 1th cursor
        let cursor = source.begin().advance(1);
        assert!(!source.is_eof(cursor));
        assert_eq!(source.rest_len(cursor), 3);
        assert_eq!(
            source.take_while(cursor, |c| c == "a"),
            Some(Cursor::from(2))
        );
        assert_eq!(source.take_while(cursor, |c| c == "b"), None);
        assert_eq!(source.take_n(cursor, 0), None);
        assert_eq!(source.take_n(cursor, 1), Some((cursor.advance(1), "a")));
        assert_eq!(source.take_n(cursor, 3), Some((cursor.advance(3), "abc")));
        assert_eq!(source.take_n(cursor, 4), None);

        let full_len = source.rest_len(source.begin());
        assert_eq!(full_len, 4);

        // end cursor
        let cursor = source.end();
        assert!(source.is_eof(cursor));
        assert_eq!(source.rest_len(cursor), 0);
        assert_eq!(source.take_while(cursor, |c| c == "a"), None);
        assert_eq!(source.take_while(cursor, |c| c == "b"), None);
        assert_eq!(source.take_n(cursor, 0), None);
        assert_eq!(source.take_n(cursor, 1), None);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SourceLocation<'i> {
    source_file: &'i SourceFile<'i>,
    range: (Cursor, Cursor),
}

#[derive(Debug, Default, Clone, Copy, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExprIndex(usize);

impl ExprIndex {
    pub const fn from(n: usize) -> Self {
        Self(n)
    }

    pub const fn get(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct ExprCtx<'i> {
    loc: SourceLocation<'i>,
    args: Vec<ExprIndex>,
}

impl<'i> ExprCtx<'i> {
    fn new(source_file: &'i SourceFile<'i>, begin: Cursor) -> Self {
        Self {
            loc: SourceLocation {
                source_file,
                range: (begin, begin),
            },
            args: vec![],
        }
    }

    fn add_arg(&mut self, idx: ExprIndex) {
        self.args.push(idx);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PrimitiveCtx<'i, T> {
    loc: SourceLocation<'i>,
    v: T,
}

impl<'i, T> PrimitiveCtx<'i, T> {
    fn new(source_file: &'i SourceFile<'i>, range: (Cursor, Cursor), v: T) -> Self {
        Self {
            loc: SourceLocation { source_file, range },
            v,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr<'i> {
    List(ExprCtx<'i>),
    Unit(ExprCtx<'i>),
    Define(ExprCtx<'i>),
    Lambda(ExprCtx<'i>),
    FunctionCall(ExprCtx<'i>),
    Name(PrimitiveCtx<'i, &'i str>),
    Integer(PrimitiveCtx<'i, i64>),
    Symbol(PrimitiveCtx<'i, Symbol>),
}

impl<'i> Expr<'i> {
    fn add_arg(&mut self, idx: ExprIndex) {
        match self {
            Expr::List(c) => c.add_arg(idx),
            Expr::Define(c) => c.add_arg(idx),
            Expr::Lambda(c) => c.add_arg(idx),
            Expr::FunctionCall(c) => c.add_arg(idx),
            _ => unreachable!("only Expr types can have args"),
        }
    }

    fn set_end_cursor(&mut self, end: Cursor) {
        match self {
            Expr::List(c) => c.loc.range.1 = end,
            Expr::Unit(c) => c.loc.range.1 = end,
            Expr::Define(c) => c.loc.range.1 = end,
            Expr::Lambda(c) => c.loc.range.1 = end,
            Expr::FunctionCall(c) => c.loc.range.1 = end,
            Expr::Name(c) => c.loc.range.1 = end,
            Expr::Integer(c) => c.loc.range.1 = end,
            Expr::Symbol(c) => c.loc.range.1 = end,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SyntaxError<'i> {
    loc: SourceLocation<'i>,
    description: String,
}

impl<'i> std::fmt::Display for SyntaxError<'i> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<'i> From<ParseCharError<'i>> for SyntaxError<'i> {
    fn from(value: ParseCharError<'i>) -> Self {
        SyntaxError {
            loc: SourceLocation {
                source_file: value.source_file,
                range: (value.got, value.got.advance(1)),
            },
            description: value.to_string(),
        }
    }
}

impl<'i> From<ParseIdentifierError<'i>> for SyntaxError<'i> {
    fn from(value: ParseIdentifierError<'i>) -> Self {
        SyntaxError {
            loc: SourceLocation {
                source_file: value.source_file,
                range: (value.got, value.got.advance(1)),
            },
            description: value.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct Ast<'i> {
    exprs: Vec<Expr<'i>>,
}

impl<'i> Default for Ast<'i> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'i> Ast<'i> {
    fn new() -> Self {
        Self {
            exprs: Vec::with_capacity(3000),
        }
    }

    fn get_expr_at(&self, expr_idx: ExprIndex) -> Option<&Expr<'i>> {
        if expr_idx.get() < self.exprs.len() {
            Some(&self.exprs[expr_idx.get()])
        } else {
            None
        }
    }

    fn add_expr(&mut self, expr: Expr<'i>) -> ExprIndex {
        self.exprs.push(expr);
        ExprIndex::from(self.exprs.len() - 1)
    }

    fn parse_expr(
        &mut self,
        source_file: &'i SourceFile<'i>,
        cursor: Cursor,
    ) -> Result<(Expr<'i>, Cursor), SyntaxError<'i>> {
        let lparen_begin = skip_spaces(source_file, cursor);

        let (lparen_end, _) = lparen(source_file, lparen_begin)?;

        let begin = skip_spaces(source_file, lparen_end);

        // ()
        if let Ok((rparen_end, ())) = rparen(source_file, begin) {
            return Ok((
                Expr::Unit(ExprCtx {
                    loc: SourceLocation {
                        source_file,
                        range: (lparen_begin, rparen_end),
                    },
                    args: vec![],
                }),
                rparen_end,
            ));
        }

        if let Ok((id_end, id)) = identifier(source_file, begin) {
            match id {
                // (define ...)
                "define" => {
                    let (expr, end) = self.parse_define(source_file, id_end, lparen_begin)?;
                    return Ok((expr, end));
                }
                // (lambda ...)
                "lambda" => {
                    let (expr, end) = self.parse_lambda(source_file, id_end, lparen_begin)?;
                    return Ok((expr, end));
                }
                // (call ...)
                name => {
                    let (expr, end) =
                        self.parse_function_call(source_file, id_end, lparen_begin, name)?;
                    return Ok((expr, end));
                }
            }
        }

        // (list ...)
        if let Ok((list_end, _)) = lparen(source_file, begin) {
            let (expr, end) = self.parse_list(source_file, list_end)?;
            return Ok((expr, end));
        }

        // {integer}
        if let Ok((end, v)) = integer(source_file, begin) {
            return Ok((
                Expr::Integer(PrimitiveCtx::new(source_file, (begin, end), v)),
                end,
            ));
        }

        // {symbol}
        if let Ok((end, v)) = symbol(source_file, begin) {
            return Ok((
                Expr::Symbol(PrimitiveCtx::new(source_file, (begin, end), v)),
                end,
            ));
        }

        Err(ParseCharError {
            source_file,
            got: begin,
            expected: "an unit/define/lambda/call/list".to_owned(),
        }
        .into())
    }

    fn parse_lambda_args(
        &mut self,
        source_file: &'i SourceFile<'i>,
        cursor: Cursor,
    ) -> Result<(Expr<'i>, Cursor), SyntaxError<'i>> {
        let mut expr = Expr::List(ExprCtx::new(source_file, cursor));
        let mut cc = cursor;
        loop {
            let begin = skip_spaces(source_file, cc);
            if let Ok((end, ())) = rparen(source_file, begin) {
                expr.set_end_cursor(end);
                return Ok((expr, end));
            }
            let (id_end, id) = identifier(source_file, begin)?;
            expr.add_arg(self.add_expr(Expr::Name(PrimitiveCtx::new(
                source_file,
                (begin, id_end),
                id,
            ))));
            cc = id_end;
        }
    }

    fn parse_lambda(
        &mut self,
        source_file: &'i SourceFile<'i>,
        cursor: Cursor,
        lparen_begin: Cursor,
    ) -> Result<(Expr<'i>, Cursor), SyntaxError<'i>> {
        let args_begin = skip_spaces(source_file, cursor);
        let (args, args_end) = self.parse_lambda_args(source_file, args_begin)?;
        let body_begin = skip_spaces(source_file, args_end);
        let (body, body_end) = self.parse_expr(source_file, body_begin)?;
        let (rparen_end, _) = rparen(source_file, body_end)?;

        let expr = Expr::Lambda(ExprCtx {
            loc: SourceLocation {
                source_file,
                range: (lparen_begin, rparen_end),
            },
            args: vec![self.add_expr(args), self.add_expr(body)],
        });

        Ok((expr, rparen_end))
    }

    fn parse_define(
        &mut self,
        source_file: &'i SourceFile<'i>,
        cursor: Cursor,
        lparen_begin: Cursor,
    ) -> Result<(Expr<'i>, Cursor), SyntaxError<'i>> {
        let (id_end, id) = identifier(source_file, cursor)?;
        let name = Expr::Name(PrimitiveCtx::new(source_file, (cursor, id_end), id));
        let expr_begin = skip_spaces(source_file, id_end);
        let (expr, expr_end) = self.parse_expr(source_file, expr_begin)?;
        let (rparen_end, _) = rparen(source_file, expr_end)?;

        let expr = Expr::Define(ExprCtx {
            loc: SourceLocation {
                source_file,
                range: (lparen_begin, rparen_end),
            },
            args: vec![self.add_expr(name), self.add_expr(expr)],
        });

        Ok((expr, rparen_end))
    }

    fn parse_function_call(
        &mut self,
        source_file: &'i SourceFile<'i>,
        cursor: Cursor,
        lparen_begin: Cursor,
        name: &'i str,
    ) -> Result<(Expr<'i>, Cursor), SyntaxError<'i>> {
        let (id_end, id) = identifier(source_file, cursor)?;
        let name = Expr::Name(PrimitiveCtx::new(source_file, (cursor, id_end), id));
        let mut expr = Expr::List(ExprCtx::new(source_file, id_end));
        let mut cc = id_end;
        loop {
            let begin = skip_spaces(source_file, cc);
            if let Ok((end, ())) = rparen(source_file, begin) {
                expr.set_end_cursor(end);
                return Ok((
                    Expr::FunctionCall(ExprCtx {
                        loc: SourceLocation {
                            source_file,
                            range: (lparen_begin, end),
                        },
                        args: vec![self.add_expr(name), self.add_expr(expr)],
                    }),
                    end,
                ));
            }
            let (arg, arg_end) = self.parse_expr(source_file, begin)?;
            expr.add_arg(self.add_expr(arg));
            cc = arg_end;
        }
    }

    fn parse_list(
        &mut self,
        source_file: &'i SourceFile<'i>,
        cursor: Cursor,
    ) -> Result<(Expr<'i>, Cursor), SyntaxError<'i>> {
        let mut cc = skip_spaces(source_file, source_file.begin());
        let mut file_list_expr = Expr::List(ExprCtx::new(source_file, cc));

        loop {
            let (expr, expr_end) = self.parse_expr(source_file, cc)?;
            file_list_expr.add_arg(self.add_expr(expr));
            cc = skip_spaces(source_file, expr_end);
            if source_file.is_eof(cc) {
                file_list_expr.set_end_cursor(expr_end);
                return Ok((file_list_expr, expr_end));
            }
        }
    }

    fn parse_file(&mut self, source_file: &'i SourceFile<'i>) -> Result<Expr<'i>, SyntaxError<'i>> {
        self.parse_list(source_file, source_file.begin())
            .map(|(expr, _)| expr)
    }
}

//#[cfg(test)]
//mod parser_test {
//    use super::*;
//
//    #[test]
//    fn file_contains_unit_only() {
//        fn sf(content: &str, list_range: (Cursor, Cursor), unit_range: (Cursor, Cursor)) {
//            let source_file = SourceFile::new(content.to_owned());
//            let source_file = source_file.unicode_source_file();
//
//            let mut ast = Ast::new();
//            let r = ast.parse_file(&source_file);
//            assert!(matches!(
//                r.clone(),
//                Ok(Expr::list(ExprCtx { loc: SourceLocation {range: list_range, ..}, args}))
//                if args.len() == 1 && args[0] == ExprIndex::from(0)
//            ));
//            assert!(
//                matches!(
//                    ast.get_expr_at(ExprIndex::from(0)),
//                    Some(&Expr::Unit(ExprCtx {loc: SourceLocation {range: unit_range, ..}, ref args}))
//                    if args.is_empty()
//                )
//            );
//        }
//
//        for (content, list_range, unit_range) in [
//            (
//                "()",
//                (Cursor::from(0), Cursor::from(2)),
//                (Cursor::from(0), Cursor::from(2)),
//            ),
//            (
//                "  (  )  ",
//                (Cursor::from(2), Cursor::from(6)),
//                (Cursor::from(2), Cursor::from(6)),
//            ),
//        ] {
//            sf(content, list_range, unit_range);
//        }
//    }
//
//    #[test]
//    fn empty_file() {
//        fn sf(content: &str, list_range: (Cursor, Cursor)) {
//            let source_file = SourceFile::new(content.to_owned());
//            let source_file = source_file.unicode_source_file();
//
//            let mut ast = Ast::new();
//            let r = ast.parse_file(&source_file);
//            assert!(matches!(
//                r.clone(),
//                Ok(Expr::list(ExprCtx { loc: SourceLocation {range: list_range, ..}, args}))
//                if args.is_empty()
//            ));
//        }
//
//        for (content, list_range) in [
//            ("", (Cursor::from(0), Cursor::from(0))),
//            ("  ", (Cursor::from(2), Cursor::from(2))),
//        ] {
//            sf(content, list_range);
//        }
//    }
//
//    #[test]
//    fn test_define() {
//        let source_file = SourceFile::new("(define a 1)".to_owned());
//        let source_file = source_file.unicode_source_file();
//
//        let mut ast = Ast::new();
//        let r = ast.parse_file(&source_file);
//        assert!(matches!(
//            r.clone(),
//            Ok(Expr::list(ExprCtx { loc: SourceLocation {range: (Cursor(0), Cursor(11)), ..}, args}))
//            if args.len() == 1
//        ));
//        assert!(matches!(
//            ast.get_expr_at(ExprIndex(0)),
//            Some(&Expr::Define(ExprCtx {
//                loc: SourceLocation {
//                    range: (Cursor(0), Cursor(11)),
//                    ..
//                },
//                ref args
//            })) if args.len() == 2
//        ));
//    }
//}
