#![allow(unused)]

use std::{borrow::Cow, cell::Cell};

use unicode_segmentation::UnicodeSegmentation;

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
