#![allow(unused)]

#[derive(Debug, Clone, Eq)]
pub struct Source {
    content: String,
    path: std::path::PathBuf,
}

impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl Source {
    pub(crate) fn new<P: AsRef<std::path::Path>>(content: String, path: P) -> Self {
        Self {
            content,
            path: path.as_ref().to_owned(),
        }
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "path: {}", self.path.display())
    }
}

#[derive(Debug, Default, Clone, Copy, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cursor(pub usize);

impl Source {
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
    pub fn len(&self) -> usize {
        self.content.len()
    }
    pub fn take_while<P: Fn(char) -> bool>(&self, p: P) -> Option<&str> {
        if (self.is_empty()) {
            return None;
        }

        match self.content.find(|c| !p(c)) {
            Some(0) => None,
            Some(n) => Some(&self.content[..n]),
            None => Some(&self.content),
        }
    }
    pub fn take_n(&self, n: usize) -> Option<&str> {
        if self.content.len() < n || n == 0 {
            None
        } else {
            Some(&self.content[..n])
        }
    }
    pub fn char(&self) -> Option<char> {
        self.content.chars().next()
    }
}

fn is_eof(src: &Source, cursor: Cursor) -> bool {
    src.is_empty() || src.len() <= cursor.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceRange<'src> {
    pub(crate) src: &'src Source,
    pub(crate) begin: usize,
    pub(crate) end: usize,
}

impl<'src> std::fmt::Display for SourceRange<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}-{}", self.src, self.begin, self.end - 1)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token<'src> {
    Integer { v: u64, loc: SourceRange<'src> }, // signed 64-bit integer
    Float { v: f64, loc: SourceRange<'src> },
    String { v: String, loc: SourceRange<'src> },
    Identifier { v: String, loc: SourceRange<'src> },
    LParen { loc: SourceRange<'src> },
    RParen { loc: SourceRange<'src> },
}

impl<'src> std::fmt::Display for Token<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Integer { v, loc } => write!(f, "{}: `{}::Integer`", loc, v),
            Token::Float { v, loc } => write!(f, "{}: `{}::Float`", loc, v),
            Token::String { v, loc } => write!(f, "{}: `{}::String`", loc, v),
            Token::Identifier { v, loc } => write!(f, "{}: `{}::Identifier`", loc, v),
            Token::LParen { loc } => write!(f, "{}: `LParen`", loc),
            Token::RParen { loc } => write!(f, "{}: `RParen`", loc),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fmt_token() {
        let src = Source::new(String::new(), "hi");
        let loc = SourceRange {
            src: &src,
            begin: 0,
            end: 3,
        };
        assert_eq!(
            format!("{}", Token::Integer { v: 42, loc }),
            format!("{}: `42::Integer`", loc)
        );
        assert_eq!(
            format!("{}", Token::Float { v: 42.42, loc }),
            format!("{}: `42.42::Float`", loc)
        );
        assert_eq!(
            format!(
                "{}",
                Token::String {
                    v: "hello world".to_owned(),
                    loc
                }
            ),
            format!("{}: `hello world::String`", loc)
        );
        assert_eq!(
            format!(
                "{}",
                Token::Identifier {
                    v: "foo".to_owned(),
                    loc
                }
            ),
            format!("{}: `foo::Identifier`", loc)
        );
        assert_eq!(
            format!("{}", Token::LParen { loc }),
            format!("{}: `LParen`", loc)
        );
        assert_eq!(
            format!("{}", Token::RParen { loc }),
            format!("{}: `RParen`", loc)
        );
    }

    #[test]
    fn test_source() {
        let source = Source::new(String::new(), "hi");

        assert!(source.is_empty());
        assert!(is_eof(&source, Cursor(0)));
        assert_eq!(source.len(), 0);
        assert_eq!(source.take_while(|c| c == ' '), None);
        assert_eq!(source.take_n(0), None);
        assert_eq!(source.take_n(1), None);
        assert_eq!(source.char(), None);

        let content = "aabc";
        let source = Source::new(content.to_owned(), "hi");

        assert!(!source.is_empty());
        for i in 0..6 {
            if i < source.len() {
                assert!(!is_eof(&source, Cursor(i)));
            } else {
                assert!(is_eof(&source, Cursor(i)));
            }
        }
        assert_eq!(source.len(), 4);
        assert_eq!(source.take_while(|c| c == 'a'), Some("aa"));
        assert_eq!(source.take_while(|c| c == 'b'), None);
        assert_eq!(source.take_n(0), None);
        assert_eq!(source.take_n(1), Some("a"));
        assert_eq!(source.take_n(source.len()), Some(content));
        assert_eq!(source.take_n(source.len() + 1), None);
        assert_eq!(source.char(), Some('a'));
    }
}
