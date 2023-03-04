#![warn(future_incompatible)]
#![warn(rust_2021_compatibility)]
#![warn(missing_debug_implementations)]
#![forbid(overflowing_literals)]

mod combinators;
mod lexer;

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDocTests;
