//! ctlab rust executable

// In this binary, we may not always be using all the functions.  Rust by
// default creates a warning when code isn't used (so-called `dead code`).  This
// is fine in this lab, so we will explicitly tell the compiler to ignore dead
// code warnings.
#![allow(dead_code)]

// The `mod` command loads a particular module which is just a self-contained
// file.  So `mod object` tells the compiler the `object.rs` should be loaded.
mod integration;
mod misc;
mod object;
mod projection;

// Rust uses namespaces so that if two crates or modules define the same
// function, they don't overwrite each other.  In order to use a function, you
// can use the full namespace path; however, the `use` commands allows for the
// particular definition to be used more easily.  For example, one could have
//
// ```
// let value1 = my_crate::some::long::path::foo(5, 4);
// let value2 = my_crate::some::long::path::foo(2, 3);
// ```
//
// but this is very long a tedious so the `use` statement allows for:
//
// ```
// use my_crate::some::long::path::foo;
//
// let value1 = foo(5, 4);
// let value2 = foo(2, 3);
// ```
use misc::*;
use ndarray::prelude::*;
use num::Complex;
use object::{AttenuationObject, ObjectCollection, Rectangle};
use projection::Projection;
use std::f64::consts::PI;


fn main() {
    // 2-Square Reference Image
    ////////////////////////////////////////
    println!("Squares");
    let rectangle1 = Rectangle::new((-1., 0.), (-1., 0.), 1.);
    let rectangle2 = Rectangle::new((0., 1.), (0., 1.), 1.);

    let mut collection = ObjectCollection::new();
    collection.push(&rectangle1);
    collection.push(&rectangle2);

    let img = array_to_img(&collection.to_array(
        &Array1::linspace(-2.0, 2.0, 1024),
        &Array1::linspace(-2.0, 2.0, 1024),
    ));
    img.save("output/squares-original.png").unwrap();

}
