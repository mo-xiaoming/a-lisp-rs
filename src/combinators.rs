#![allow(unused)]

#[derive(Debug)]
pub(crate) struct SourceRange {
    begin: usize,
    end: usize,
}

#[derive(Debug)]
pub(crate) enum ParseError<'src> {
    Unexpected {
        src: &'src Source,
        expected: String,
        got: SourceRange,
    },
    EarlyEof {
        src: &'src Source,
        expected: String,
    },
}

impl<'src> std::fmt::Display for ParseError<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Unexpected { src, expected, got } => write!(
                f,
                "expected `{}`, got `{}`",
                expected,
                &src.content[got.begin..=got.end]
            ),
            ParseError::EarlyEof { src, expected } => {
                write!(f, "expected `{}`, but reached the end of file", expected)
            }
        }
    }
}

impl<'src> std::error::Error for ParseError<'src> {}

#[derive(Debug, Default, Clone, Copy, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Cursor {
    r: usize,
    c: usize,
    off: usize,
}

type IResult<'src, O> = Result<(Cursor, O), ParseError<'src>>;

#[derive(Debug)]
pub(crate) struct Source {
    content: String,
}

fn char(c: char) -> impl Fn(&Source, Cursor) -> IResult<'_, char> {
    move |src: &Source, mut cursor: Cursor| {
        let x = src
            .content
            .chars()
            .next()
            .ok_or_else(|| ParseError::EarlyEof {
                src,
                expected: c.to_string(),
            })?;

        if x == c {
            cursor.c += 1;
            cursor.off += 1;
            Ok((cursor, x))
        } else {
            Err(ParseError::Unexpected {
                src,
                expected: c.to_string(),
                got: SourceRange {
                    begin: cursor.off,
                    end: cursor.off + 1,
                },
            })
        }
    }
}

pub(crate) fn lparen() -> impl Fn(&Source, Cursor) -> IResult<'_, char> {
    move |src: &Source, mut cursor: Cursor| char('(')(src, cursor)
}

pub(crate) fn rparen() -> impl Fn(&Source, Cursor) -> IResult<'_, char> {
    move |src: &Source, mut cursor: Cursor| char(')')(src, cursor)
}

pub(crate) fn integer() -> impl Fn(&Source, Cursor) -> IResult<'_, i64> {
    move |src: &Source, mut cursor: Cursor| {
        let ef = || String::from("{integer}");

        if src.content.is_empty() {
            return Err(ParseError::EarlyEof {
                src,
                expected: ef(),
            });
        }

        let s = src
            .content
            .chars()
            .take_while(|c| c.is_numeric())
            .collect::<String>();
        if s.is_empty() {
            return Err(ParseError::Unexpected {
                src,
                expected: ef(),
                got: SourceRange {
                    begin: cursor.off,
                    end: cursor.off + 1,
                },
            });
        }

        Ok((
            Cursor {
                r: cursor.r,
                c: cursor.c + s.len(),
                off: cursor.off + s.len(),
            },
            s.parse::<i64>().unwrap(),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn char_tests<P>(c: char, p: P)
    where
        P: Fn(&Source, Cursor) -> IResult<'_, char>,
    {
        let source = Source {
            content: c.to_string(),
        };
        let r = p(&source, Cursor::default())
            .unwrap_or_else(|_| panic!("`{}` should be properly accepted", c));
        assert_eq!(r, (Cursor { r: 0, c: 1, off: 1 }, c));

        let source = Source {
            content: String::new(),
        };
        let r = p(&source, Cursor::default());
        assert!(
            matches!(r, Err(ParseError::EarlyEof { expected,.. }) if expected == c.to_string())
        );

        let source = Source {
            content: "x".to_owned(),
        };
        let r = p(&source, Cursor::default());
        assert!(
            matches!(r, Err(ParseError::Unexpected { expected, got: SourceRange { begin, end }, .. }) if expected == c.to_string() && begin == 0 && end == 1)
        );
    }

    #[test]
    fn test_char() {
        char_tests('c', char('c'));
    }

    #[test]
    fn test_parens() {
        char_tests('(', lparen());
        char_tests(')', rparen());
    }
}
