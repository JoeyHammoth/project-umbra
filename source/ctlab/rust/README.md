# Computed Tomography Lab -- Rust

[Rust](https://www.rust-lang.org) is a modern general purpose compiled
programming language designed to be safe, concurrent and practical.  It is quite
similar to C/C++, but has many of the nice features from modern programming
language such as Python.

## Installing the Rust compiler

The Rust compiler will (generally) not be installed on your computer;
fortunately, the Rust compiler is free and open-source and can be easily
installed.  In particular, on Linux all you need to do is copy-paste the
following line into the terminal:

```sh
curl https://sh.rustup.rs -sSf | sh
```

If you want to install Rust on Windows, you can obtain an installer
from [their website](https://www.rust-lang.org).

You can verify that Rust has been correctly installed by ensuring that the
following commands work (note that you may have to close then reopen your
terminal).

```sh
cargo --version
rustc --version
```

The following couple of steps add a few niceties (which I would highly
recommend), but are not necessary.  They will install a Rust auto-completer
called `racer`, and a Rust formatter called `rustfmt`, and then help you
configure Atom to work with them.

```sh
cargo install racer
rustup install nightly
rustup run nightly cargo install rustfmt-nightly
mkdir -p ~/src
git clone --branch stable --depth 1 https://github.com/rust-lang/rust.git ~/src/rust
```

While the above commands are running, open the Atom text editor and navigate to
Settings -> Install.  Then search for, and install, the following packages:

- `language-rust`
- `linter-rust`
- `rustfmt`
- `racer`

They should automatically find the appropriate tools, though you'll probably
need to log-out and log back in.

## Getting Started

The lab comes with a complete skeleton code which you are meant to fill out.  In
particular, functions which you need to implement initially contain only the line:

```rust
unimplemented!()
```

which raises an error if you try and run the function.

The first thing you should do is familiarize yourself with what code is already
there.  You don't need to understand the every minute detail of every function,
but you should have a rough idea of what functions do what, and what is out
there.

One of the great features of Rust is that it is self-documenting.  To see this,
open up your terminal and in the `rust` directory, type:

```sh
cargo doc --open
```

This will create all the documentation, and then open your browser to it.

The syntax for Rust was designed to be fairly intuitive, but in any case you can
always refer to the following online book for a few examples.

> [The Rust Programming Language](https://doc.rust-lang.org/book/second-edition/)

Having said that, there should hopefully be enough code already present in the
skeleton to allow you to get started.

In order to run your code, you need to run:

```sh
cargo run
```

This will compile everything and run the executable, but it performs no
optimizations; as a result, the executable can be quite slow.  If you want to
optimize the executable, then run

```sh
cargo run --release
```

The trade-off is that the compilation step will take a lot longer, but the
executable can be orders of magnitude faster.
