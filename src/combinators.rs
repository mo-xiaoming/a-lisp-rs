#![allow(unused)]

use crate::lexer::Cursor;

use crate::lexer::{Source, SourceRange, Token};

pub(crate) trait ParseError<'src> {
    fn description(&'src self, src: &'src Source) -> String;
}

type IResult<'src, O, E> = Result<(Cursor, O), E>;

#[derive(Debug, Clone)]
struct EarlyEof<'src> {
    src: &'src Source,
    expected: String,
}

impl<'src> ParseError<'src> for EarlyEof<'src> {
    fn description(&'src self, src: &'src Source) -> String {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct ParseCharError<'src> {
    expected: String,
    got: SourceRange<'src>,
}

impl<'src> ParseError<'src> for ParseCharError<'src> {
    fn description(&'src self, src: &'src Source) -> String {
        todo!()
    }
}

fn char(c: char) -> impl Fn(&Source, Cursor) -> IResult<'_, (char, SourceRange), ParseCharError> {
    move |src: &Source, mut cursor: Cursor| {
        assert!(!src.is_empty(), "char parser doesn't support empty source");
        assert!(
            !src.is_empty() && src.len() > cursor.0,
            "char parser reaches EOF before parsing"
        );

        let x = src.char().unwrap();

        if x == c {
            cursor.0 += 1;
            Ok((
                cursor,
                (
                    x,
                    SourceRange {
                        src,
                        begin: cursor.0 - 1,
                        end: cursor.0,
                    },
                ),
            ))
        } else {
            Err(ParseCharError {
                expected: c.to_string(),
                got: SourceRange {
                    src,
                    begin: cursor.0,
                    end: cursor.0 + 1,
                },
            })
        }
    }
}

pub fn lparen() -> impl Fn(&Source, Cursor) -> IResult<'_, Token, ParseCharError> {
    move |src: &Source, mut cursor: Cursor| {
        char('(')(src, cursor).map(|(i, (_, loc))| (i, Token::LParen { loc }))
    }
}

pub fn rparen() -> impl Fn(&Source, Cursor) -> IResult<'_, Token, ParseCharError> {
    move |src: &Source, mut cursor: Cursor| {
        char(')')(src, cursor).map(|(i, (_, loc))| (i, Token::RParen { loc }))
    }
}

#[derive(Debug, Clone)]
pub enum ParseIntegerError<'src> {
    Unexpected { got: SourceRange<'src> },
    Overflow { got: SourceRange<'src> },
    LeadingZero { got: SourceRange<'src> },
}

pub fn integer() -> impl Fn(&Source, Cursor) -> IResult<'_, Token, ParseIntegerError> {
    move |src: &Source, mut cursor: Cursor| {
        assert!(
            !src.is_empty(),
            "integer parser doesn't support empty source"
        );

        if src.char() == Some('0') {
            return Err(ParseIntegerError::LeadingZero {
                got: SourceRange {
                    src,
                    begin: cursor.0,
                    end: cursor.0 + 1,
                },
            });
        }

        let s = src.take_while(|c| c.is_ascii_digit()).ok_or({
            ParseIntegerError::Unexpected {
                got: SourceRange {
                    src,
                    begin: cursor.0,
                    end: cursor.0 + 1,
                },
            }
        })?;

        let i = s.parse::<u64>().map_err(|e| match e.kind() {
            std::num::IntErrorKind::PosOverflow => ParseIntegerError::Overflow {
                got: SourceRange {
                    src,
                    begin: cursor.0,
                    end: cursor.0 + s.len(),
                },
            },
            _ => unreachable!("unexpected {{integer}} parsing error"),
        })?;

        Ok((
            Cursor(cursor.0 + s.len()),
            Token::Integer {
                v: i,
                loc: SourceRange {
                    src,
                    begin: cursor.0,
                    end: cursor.0 + s.len(),
                },
            },
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParseIdentifierError<'src> {
    got: SourceRange<'src>,
}

pub fn identifier() -> impl Fn(&Source, Cursor) -> IResult<'_, Token, ParseIdentifierError> {
    move |src: &Source, mut cursor: Cursor| {
        assert!(
            !src.is_empty(),
            "identifier parser doesn't support empty source"
        );

        let ef = || ParseIdentifierError {
            got: SourceRange {
                src,
                begin: cursor.0,
                end: cursor.0 + 1,
            },
        };

        let s = src
            .take_while(|c| c.is_ascii_alphanumeric() || c == '_')
            .ok_or_else(ef)?;
        s.chars()
            .next()
            .filter(|c| !c.is_ascii_digit())
            .ok_or_else(ef)?;

        Ok((
            Cursor(cursor.0 + s.len()),
            Token::Identifier {
                v: s.to_owned(),
                loc: SourceRange {
                    src,
                    begin: cursor.0,
                    end: cursor.0 + s.len(),
                },
            },
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic(expected = "char parser doesn't support empty source")]
    fn lparen_panic_on_empty_source1() {
        let source = Source::new(String::new(), "hi");
        let _r = lparen()(&source, Cursor::default());
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF before parsing")]
    fn lparen_panic_on_empty_source2() {
        let content = "hi".to_owned();
        let source = Source::new(content, "hi");
        let _r = lparen()(&source, Cursor(source.len()));
    }

    #[test]
    #[should_panic(expected = "char parser doesn't support empty source")]
    fn rparen_panic_on_empty_source_1() {
        let source = Source::new(String::new(), "hi");
        let _r = rparen()(&source, Cursor::default());
    }

    #[test]
    #[should_panic(expected = "char parser reaches EOF before parsing")]
    fn rparen_panic_on_empty_source2() {
        let content = "hi".to_owned();
        let source = Source::new(content, "hi");
        let _r = rparen()(&source, Cursor(source.len()));
    }

    fn test_paren_ok<P, T>(parser: &P, c: char, content: &str, t: &T)
    where
        P: Fn(&Source, Cursor) -> IResult<'_, Token, ParseCharError>,
        T: for<'src> Fn(&'src Source) -> Token<'src>,
    {
        let source = Source::new(content.to_owned(), "hi");
        let r = parser(&source, Cursor::default());
        assert!(matches!(
            r,
            Ok((
                Cursor(1),
                token,
            )) if token == t(&source)
        ));
    }

    fn test_paren_err<P>(parser: &P, c: char)
    where
        P: Fn(&Source, Cursor) -> IResult<'_, Token, ParseCharError>,
    {
        let source = Source::new("x".to_owned(), "hi");
        let r = parser(&source, Cursor::default());
        assert!(
            matches!(r, Err(ParseCharError{ expected, got: SourceRange {  begin: 0, end: 1,.. } }) if expected == c.to_string() )
        );
    }

    #[test]
    fn test_parens() {
        let p = lparen();
        fn lp(src: &Source) -> Token {
            Token::LParen {
                loc: SourceRange {
                    src,
                    begin: 0,
                    end: 1,
                },
            }
        };

        test_paren_ok(&p, '(', "(", &lp);
        test_paren_ok(&p, '(', "(xx", &lp);
        test_paren_err(&p, '(');

        let p = rparen();
        fn rp(src: &Source) -> Token {
            Token::RParen {
                loc: SourceRange {
                    src,
                    begin: 0,
                    end: 1,
                },
            }
        };

        test_paren_ok(&p, ')', ")", &rp);
        test_paren_ok(&p, ')', ")xx", &rp);
        test_paren_err(&p, ')');
    }

    #[test]
    #[should_panic(expected = "integer parser doesn't support empty source")]
    fn integer_panics_on_empty_source() {
        let source = Source::new(String::new(), "hi");
        let _r = integer()(&source, Cursor::default());
    }

    #[test]
    fn test_integer() {
        let sf = |(s, len): (&str, usize)| {
            let source = Source::new(s.to_owned(), "hi");
            let r = integer()(&source, Cursor::default());
            assert!(matches!(
                r,
                Ok((
                    Cursor(off),
                    Token::Integer {
                        v: 12345,
                        loc: SourceRange {
                            begin: 0,
                            end,
                            ..
                        }
                    }
                )) if off == 5 && end == 5
            ));
        };

        for s in [("12345", 5), ("12345abc", 5)] {
            sf(s);
        }

        let source = Source::new("012345".to_owned(), "hi");
        let r = integer()(&source, Cursor::default());
        assert!(matches!(
            r,
            Err(ParseIntegerError::LeadingZero {
                got: SourceRange {
                    begin: 0,
                    end: 1,
                    ..
                }
            })
        ));

        let source = Source::new("abc".to_owned(), "hi");
        let r = integer()(&source, Cursor::default());
        assert!(matches!(
            r,
            Err(ParseIntegerError::Unexpected {
                got: SourceRange {
                    begin: 0,
                    end: 1,
                    ..
                }
            })
        ));

        let ov = "18446744073709551616".to_owned(); // u64::max + 1
        let source = Source::new(ov.clone(), "hi");
        let r = integer()(&source, Cursor::default());
        assert!(matches!(
            r,
            Err(ParseIntegerError::Overflow {
                got: SourceRange {
                    begin: 0,
                    end,
                    ..
                }
            }) if end == ov.len()
        ));
    }

    #[test]
    #[should_panic(expected = "identifier parser doesn't support empty source")]
    fn identifier_panics_on_empty_source() {
        let source = Source::new(String::new(), "hi");
        let _r = identifier()(&source, Cursor::default());
    }

    #[test]
    fn test_identifier() {
        let sf = |s: &str| {
            let source = Source::new(s.to_owned(), "hi");
            let r = identifier()(&source, Cursor::default());
            assert!(matches!(
                r,
                Ok((
                    Cursor(off),
                    Token::Identifier {
                        v,
                        loc: SourceRange {
                            begin: 0,
                            end,
                            ..
                        }
                    }
                )) if off == s.len() && end == s.len() && v == s
            ));
        };

        for s in ["abc", "_ab", "_3a", "a_b_c", "a_3", "a3_", "__", "_", "__a"] {
            sf(s);
        }

        let ef = |s: &str| {
            let source = Source::new(s.to_owned(), "hi");
            let r = identifier()(&source, Cursor::default());
            assert!(matches!(
                r,
                Err(ParseIdentifierError {
                    got: SourceRange {
                        begin: 0,
                        end: 1,
                        ..
                    }
                })
            ));
        };

        for s in ["3ab", "3_ab", "#ab", "@ab"] {
            ef(s);
        }
    }
}
