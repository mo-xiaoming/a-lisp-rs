#![allow(unused)]

use std::borrow::Cow;
use unicode_segmentation::UnicodeSegmentation;

use crate::combinators::{
    identifier, integer, lparen, rparen, skip_spaces, symbol, ParseCharError, ParseIdentifierError,
};

#[derive(Debug, Clone, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawInput {
    content: String,
    path: std::path::PathBuf,
}

impl RawInput {
    #[cfg(test)]
    pub(crate) fn new(content: String) -> Self {
        Self {
            content,
            path: std::path::PathBuf::from("mock"),
        }
    }

    pub(crate) fn unicode_input(&self) -> Input {
        Input {
            raw: self,
            ucs: self.content.graphemes(true).collect(),
        }
    }
}

#[derive(Debug, Clone, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Input<'i> {
    raw: &'i RawInput,
    ucs: Vec<&'i str>,
}

impl<'i> Input<'i> {
    pub(crate) fn path(&self) -> Cow<str> {
        self.raw.path.as_os_str().to_string_lossy()
    }
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

impl<'i> Input<'i> {
    pub(crate) fn begin(&self) -> Cursor {
        Cursor::from(0)
    }

    pub(crate) fn end(&self) -> Cursor {
        Cursor::from(self.ucs.len())
    }

    pub(crate) fn get_str_ref(&self, begin: Cursor, end: Cursor) -> Option<&str> {
        if begin >= end || end > self.end() {
            None
        } else {
            Some(&self.raw.content[self.distance_from_begin(begin)..self.distance_from_begin(end)])
        }
    }

    fn distance_from_begin(&self, cursor: Cursor) -> usize {
        self.raw.content[cursor.get()..].as_ptr() as usize - self.raw.content.as_ptr() as usize
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
mod test {
    use super::*;

    #[test]
    fn test_empty_input() {
        // empty source
        let empty = RawInput::new(String::new());
        let empty = empty.unicode_input();

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
    fn test_non_empty_input() {
        // non-empty source
        let source = RawInput::new("aabc".to_owned());
        let source = source.unicode_input();

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

#[derive(Debug, Clone, Copy, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct SourceLocation<'i> {
    input: &'i Input<'i>,
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
    fn new(input: &'i Input<'i>, begin: Cursor) -> Self {
        Self {
            loc: SourceLocation {
                input,
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

#[derive(Debug, Clone)]
pub enum Expr<'i> {
    Scope(ExprCtx<'i>),
    Unit(ExprCtx<'i>),
    Define(ExprCtx<'i>),
    Lambda(ExprCtx<'i>),
    FunctionCall(ExprCtx<'i>),
    Integer(PrimitiveCtx<'i, i64>),
    Symbol(PrimitiveCtx<'i, crate::combinators::Symbol>),
}

impl<'i> Expr<'i> {
    pub(crate) fn add_arg(&mut self, idx: ExprIndex) {
        match self {
            Expr::Scope(c) => c.add_arg(idx),
            Expr::Define(c) => c.add_arg(idx),
            Expr::Lambda(c) => c.add_arg(idx),
            Expr::FunctionCall(c) => c.add_arg(idx),
            _ => unreachable!("only Expr types can have args"),
        }
    }

    pub(crate) fn set_end_cursor(&mut self, end: Cursor) {
        match self {
            Expr::Scope(c) => c.loc.range.1 = end,
            Expr::Unit(c) => c.loc.range.1 = end,
            Expr::Define(c) => c.loc.range.1 = end,
            Expr::Lambda(c) => c.loc.range.1 = end,
            Expr::FunctionCall(c) => c.loc.range.1 = end,
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
                input: value.input,
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
                input: value.input,
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
        input: &'i Input<'i>,
        cursor: Cursor,
    ) -> Result<(Expr<'i>, Cursor), SyntaxError<'i>> {
        let lparen_begin = skip_spaces(input, cursor);

        let (lparen_end, _) = lparen(input, lparen_begin)?;

        let id_begin = skip_spaces(input, lparen_end);

        // ()
        if let Ok((rparen_end, ())) = rparen(input, id_begin) {
            return Ok((
                Expr::Unit(ExprCtx {
                    loc: SourceLocation {
                        input,
                        range: (lparen_begin, rparen_end),
                    },
                    args: vec![],
                }),
                rparen_end,
            ));
        }

        // ((..
        if lparen(input, id_begin).is_ok() {
            let mut expr = Expr::Scope(ExprCtx::new(input, id_begin));
            let mut c = id_begin;
            while let Ok((arg, arg_end)) = self.parse_expr(input, c) {
                c = arg_end;
                expr.add_arg(self.add_expr(arg));
                c = skip_spaces(input, c);
            }
            expr.set_end_cursor(c);
            return Ok((expr, c));
        }

        // (xxx ...)
        let (id_end, id) = identifier(input, id_begin)?;
        let tmp_expr_ctx = ExprCtx::new(input, lparen_begin);
        let mut expr = if id == "lambda" {
            Expr::Lambda(tmp_expr_ctx.clone())
        } else if id == "define" {
            Expr::Define(tmp_expr_ctx.clone())
        } else {
            Expr::FunctionCall(tmp_expr_ctx.clone())
        };

        loop {
            let mut args_begin = skip_spaces(input, id_end);

            if let Ok((arg_end, v)) = integer(input, args_begin) {
                let arg = Expr::Integer(PrimitiveCtx {
                    loc: SourceLocation {
                        input,
                        range: (args_begin, arg_end),
                    },
                    v,
                });
                args_begin = arg_end;
                expr.add_arg(self.add_expr(arg));
            } else if let Ok((arg_end, v)) = symbol(input, args_begin) {
                let arg = Expr::Symbol(PrimitiveCtx {
                    loc: SourceLocation {
                        input,
                        range: (args_begin, arg_end),
                    },
                    v,
                });
                args_begin = arg_end;
                expr.add_arg(self.add_expr(arg));
            } else if let Ok((arg, arg_end)) = self.parse_expr(input, args_begin) {
                args_begin = arg_end;
                expr.add_arg(self.add_expr(arg));
            } else if let Ok((arg_end, ())) = rparen(input, args_begin) {
                expr.set_end_cursor(arg_end);
                return Ok((expr, arg_end));
            } else {
                return Err(SyntaxError {
                    loc: SourceLocation {
                        input,
                        range: (lparen_begin, args_begin),
                    },
                    description: "expected an expression/int/symbol follows this".to_owned(),
                });
            }
        }
    }

    fn parse_file(&mut self, input: &'i Input<'i>) -> Result<Expr<'i>, SyntaxError<'i>> {
        let mut file_scope_expr = Expr::Scope(ExprCtx::new(input, input.begin()));

        let mut cc = input.begin();
        loop {
            cc = skip_spaces(input, cc);
            if input.is_eof(cc) {
                break;
            }

            let (expr, expr_end) = self.parse_expr(input, cc)?;
            file_scope_expr.add_arg(self.add_expr(expr));
            cc = expr_end;
        }

        file_scope_expr.set_end_cursor(cc);
        Ok(file_scope_expr)
    }
}

#[cfg(test)]
mod test_parse {
    use super::*;

    #[test]
    fn file_level_empty_parens() {
        let input = RawInput::new("()".to_owned());
        let input = input.unicode_input();

        let mut ast = Ast::new();
        let r = ast.parse_file(&input);
        assert!(matches!(
            r.clone(),
            Ok(Expr::Scope(ExprCtx { loc: SourceLocation {range: (Cursor(0), Cursor(2)), ..}, args}))
            if args.len() == 1 && args[0] == ExprIndex::from(0)
        ));
        assert!(
            matches!(
                ast.get_expr_at(ExprIndex::from(0)),
                Some(&Expr::Unit(ExprCtx {loc: SourceLocation {range: (Cursor(0), Cursor(2)), ..}, ref args}))
                if args.is_empty()
            ),
            "{:#?}",
                ast.get_expr_at(ExprIndex::from(0)),
        )
    }
}
