// This lib file is only used for the documentation.  Have a look at `main.rs`
// instead for the entry point of the executable.

//! The different functions and structs used in the CT lab are stored in the
//! different modules below.  By splitting up the code into multiple files, each
//! file becomes a lot more manageable which makes it easier to debug.
//!
//! When programming, a good philosophy to keep in mind when writing functions
//! is:
//!
//! > Do one thing, and do that well.
//!
//! If your functions are several hundred lines long and do a bunch of different
//! tasks, then if anything goes wrong it becomes very hard to debug.  On the
//! other hand, if each function does a very specific task, it is much easier to
//! debug.
//!
//! ## Closures
//!
//! In this lab, you will find it useful to use
//! [*closures*](https://doc.rust-lang.org/book/second-edition/ch13-01-closures.html).
//! They provide a convenient way of defining functions that will be used
//! (usually) only once.  In Rust, they are defined as:
//!
//! ```
//! let my_closure = |x, y| x * y;
//!
//! assert_eq!(my_closure(0., 0.), 0.);
//! assert_eq!(my_closure(1., 0.), 0.);
//! assert_eq!(my_closure(2., 3.), 6.);
//! ```

pub mod integration;
pub mod misc;
pub mod object;
pub mod projection;
