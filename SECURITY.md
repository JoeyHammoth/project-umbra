# Security policy

UMBRA has no published release yet. It is built from `main` and served as static
files, so the only supported version is the tip of `main`.

## Reporting a vulnerability

Please **do not** open a public issue. Use the repository's **Security** tab →
**Report a vulnerability**, which files a private advisory:

<https://github.com/JoeyHammoth/project-umbra/security/advisories/new>

Include what you did, what happened, the commit (`git rev-parse HEAD`) or the
deployed URL, and the browser and OS where it reproduces. A minimal scene file
or share URL is worth more than a description.

Reports are handled on a best-effort basis: the maintainer will confirm receipt
and give a rough timeline before discussing detail or disclosure.

## What is in scope

The app runs entirely in the browser with no backend, no accounts and no data
storage of its own, so the interesting classes are narrow:

- a malicious **scene file or share URL** that causes code execution or a
  refusal to be bypassed — `umbra_forward::scene::SceneSpec` is the only parser
  of untrusted input;
- a **dependency** with a known advisory that reaches the shipped bundle;
- a way to make the page **exfiltrate or persist** data it should not, through
  the IndexedDB cache or a share URL.

A deliberately huge scene that stalls the tab is a bug worth reporting through
the bug form, but it is not a vulnerability: the computation is local and the
browser sandbox contains it.

## What is out of scope

- Vulnerabilities in browsers, Rust, Node or GitHub itself — report those
  upstream.
- Anything requiring the victim to paste code into the console.
- Missing hardening headers on a hosting provider whose configuration lives
  outside this repository; say which host you saw and it will be raised with
  them.
