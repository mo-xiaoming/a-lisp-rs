#![allow(unused)]

#[derive(Debug)]
pub enum Token {
    Integer(i64), // signed 64-bit integer
    Float(f64),
    String(String),
    Symbol(String), // any group of characters other than an integer or parenthesis or whitespace
    LParen,
    RParen,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Integer(n) => write!(f, "{}", n),
            Token::Float(n) => write!(f, "{}", n),
            Token::String(s) => write!(f, "{}", s),
            Token::Symbol(s) => write!(f, "{}", s),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fmt_token() {
        assert_eq!(format!("{}", Token::Integer(42)), "42");
        assert_eq!(format!("{}", Token::Float(42.42)), "42.42");
        assert_eq!(
            format!("{}", Token::String("hello world".to_owned())),
            "hello world"
        );
        assert_eq!(format!("{}", Token::Symbol("foo".to_owned())), "foo");
        assert_eq!(format!("{}", Token::LParen), "(");
        assert_eq!(format!("{}", Token::RParen), ")");
    }
}
