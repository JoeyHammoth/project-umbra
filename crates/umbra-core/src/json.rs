//! A dependency-free JSON reader and writer.
//!
//! Why this exists: spec §8.1 makes scene documents JSON, and §9 makes the golden
//! dataset a committed, reviewable file. But `umbra-core` has no dependencies
//! (invariant 1) and `umbra-forward` deliberately keeps `serde_json` as a
//! test-only dependency so the wasm bundle stays lean — so a scene loader built
//! on serde would put a parser in every page load to read a file that ships with
//! the app.
//!
//! This reads the JSON that reviewed, committed documents actually contain, and
//! nothing more. The limits are deliberate and each one is an error rather than a
//! guess:
//!
//! - nesting deeper than [`MAX_DEPTH`] is rejected, so a malformed document
//!   cannot be used as a stack probe;
//! - duplicate keys in one object are rejected: "last one wins" is exactly how
//!   two readers end up disagreeing about one file;
//! - after the value only whitespace may follow, so a truncated or concatenated
//!   document fails instead of silently parsing its first half;
//! - non-finite numbers do not exist in JSON and are rejected here too.
//!
//! The writer emits two-space-indented output with one array element per line,
//! which is the same shape the Python oracle fixtures use, so regenerating a
//! committed document produces a reviewable diff rather than one long line.

use std::fmt;

/// Maximum nesting depth accepted by [`parse`].
pub const MAX_DEPTH: usize = 64;

/// A JSON value.
///
/// Numbers are `f64` because that is what the grammar describes; object keys keep
/// their document order so a rewritten file stays diffable.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// The `null` literal.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number.
    Number(f64),
    /// A string, with escapes already decoded.
    String(String),
    /// An array, in document order.
    Array(Vec<Json>),
    /// An object, in document order.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Value of `key` when this is an object, else `None`.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Entries of an object, for a loader that must reject unknown keys.
    pub fn entries(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(entries) => Some(entries),
            _ => None,
        }
    }

    /// Borrowed string contents.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(text) => Some(text),
            _ => None,
        }
    }

    /// Numeric contents.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// Boolean contents.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Array elements.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }
}

impl From<f32> for Json {
    fn from(value: f32) -> Self {
        Json::Number(f64::from(value))
    }
}

impl From<f64> for Json {
    fn from(value: f64) -> Self {
        Json::Number(value)
    }
}

impl From<u32> for Json {
    fn from(value: u32) -> Self {
        Json::Number(f64::from(value))
    }
}

impl From<u64> for Json {
    fn from(value: u64) -> Self {
        Json::Number(value as f64)
    }
}

impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Json::Bool(value)
    }
}

impl From<&str> for Json {
    fn from(value: &str) -> Self {
        Json::String(value.to_string())
    }
}

impl From<String> for Json {
    fn from(value: String) -> Self {
        Json::String(value)
    }
}

/// A parse failure, with the byte offset that caused it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonError {
    /// Byte offset into the input.
    pub offset: usize,
    /// What went wrong, in a form that can be shown to a user.
    pub message: &'static str,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for JsonError {}

/// Parses one complete JSON document.
pub fn parse(text: &str) -> Result<Json, JsonError> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let value = parser.value(0)?;
    parser.skip_whitespace();
    if parser.pos != parser.bytes.len() {
        return parser.fail("trailing characters after the document");
    }
    Ok(value)
}

/// Renders a value as indented JSON.
pub fn to_string_pretty(value: &Json) -> String {
    let mut out = String::new();
    write_value(value, 0, &mut out);
    out
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn fail<T>(&self, message: &'static str) -> Result<T, JsonError> {
        Err(JsonError {
            offset: self.pos,
            message,
        })
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\n' | b'\r' | b'\t') = self.peek() {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), JsonError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            self.fail("unexpected character")
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth > MAX_DEPTH {
            return self.fail("nesting is too deep");
        }
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => {
                self.literal("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => self.fail("expected a value"),
            None => self.fail("unexpected end of input"),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.expect(b'{')?;
        let mut entries: Vec<(String, Json)> = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(entries));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            if entries.iter().any(|(existing, _)| *existing == key) {
                return self.fail("duplicate key");
            }
            self.skip_whitespace();
            self.expect(b':')?;
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Object(entries));
                }
                _ => return self.fail("expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Array(items));
                }
                _ => return self.fail("expected ',' or ']'"),
            }
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = match self.peek() {
                Some(byte) => byte,
                None => return self.fail("unterminated string"),
            };
            self.pos += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escape = match self.peek() {
                        Some(byte) => byte,
                        None => return self.fail("unterminated escape"),
                    };
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&first) {
                                // High surrogate: the low half must follow immediately.
                                if self.peek() != Some(b'\\') {
                                    return self.fail("lone high surrogate");
                                }
                                self.pos += 1;
                                if self.peek() != Some(b'u') {
                                    return self.fail("lone high surrogate");
                                }
                                self.pos += 1;
                                let second = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&second) {
                                    return self.fail("bad low surrogate");
                                }
                                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&first) {
                                return self.fail("lone low surrogate");
                            } else {
                                first
                            };
                            match char::from_u32(code) {
                                Some(character) => out.push(character),
                                None => return self.fail("invalid code point"),
                            }
                        }
                        _ => return self.fail("unknown escape"),
                    }
                }
                byte if byte < 0x20 => return self.fail("control character in string"),
                _ => {
                    // Copy the whole UTF-8 sequence, so multi-byte characters
                    // survive and invalid encodings are rejected by `from_utf8`.
                    let start = self.pos - 1;
                    let end = start + utf8_len(byte);
                    let slice = match self.bytes.get(start..end) {
                        Some(slice) => slice,
                        None => return self.fail("truncated UTF-8 sequence"),
                    };
                    match std::str::from_utf8(slice) {
                        Ok(text) => {
                            out.push_str(text);
                            self.pos = end;
                        }
                        Err(_) => return self.fail("invalid UTF-8"),
                    }
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = match self.peek() {
                Some(byte) => byte,
                None => return self.fail("truncated \\u escape"),
            };
            self.pos += 1;
            let digit = match byte {
                b'0'..=b'9' => u32::from(byte - b'0'),
                b'a'..=b'f' => u32::from(byte - b'a') + 10,
                b'A'..=b'F' => u32::from(byte - b'A') + 10,
                _ => return self.fail("bad hex digit in \\u escape"),
            };
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let integer = self.pos;
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.pos == integer {
            return self.fail("expected a digit");
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            let fraction = self.pos;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
            if self.pos == fraction {
                return self.fail("expected a digit after '.'");
            }
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.pos += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.pos += 1;
            }
            let exponent = self.pos;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
            if self.pos == exponent {
                return self.fail("expected a digit in the exponent");
            }
        }
        match std::str::from_utf8(&self.bytes[start..self.pos]) {
            Ok(text) => match text.parse::<f64>() {
                Ok(value) if value.is_finite() => Ok(Json::Number(value)),
                _ => self.fail("number is out of range"),
            },
            Err(_) => self.fail("invalid number"),
        }
    }

    fn literal(&mut self, text: &str) -> Result<(), JsonError> {
        if self.bytes[self.pos..].starts_with(text.as_bytes()) {
            self.pos += text.len();
            Ok(())
        } else {
            self.fail("expected a literal")
        }
    }
}

fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

fn write_value(value: &Json, indent: usize, out: &mut String) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Number(number) => out.push_str(&format_number(*number)),
        Json::String(text) => write_string(text, out),
        Json::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                push_indent(indent + 1, out);
                write_value(item, indent + 1, out);
                if index + 1 < items.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            push_indent(indent, out);
            out.push(']');
        }
        Json::Object(entries) => {
            if entries.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (index, (key, item)) in entries.iter().enumerate() {
                push_indent(indent + 1, out);
                write_string(key, out);
                out.push_str(": ");
                write_value(item, indent + 1, out);
                if index + 1 < entries.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            push_indent(indent, out);
            out.push('}');
        }
    }
}

fn push_indent(level: usize, out: &mut String) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

/// Formats a number the same way twice.
///
/// Every quantity in a scene is `f32`, and writing the `f32` shortest form keeps
/// a density of 2.65 from being written as 2.6500000953674316 — which would make
/// every regeneration of a committed document a diff, and make the file
/// unreadable besides. Values that are not exactly representable as `f32` fall
/// back to the `f64` form, so nothing is lost.
fn format_number(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let narrowed = value as f32;
    if f64::from(narrowed) == value {
        if narrowed.fract() == 0.0 && narrowed.abs() < 1.0e15 {
            format!("{narrowed:.0}")
        } else {
            format!("{narrowed}")
        }
    } else {
        format!("{value}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars_arrays_and_objects() {
        let value = parse(r#"{"a": 1, "b": [true, false, null], "c": "x", "d": -2.5e2}"#)
            .expect("valid document");
        assert_eq!(value.get("a").and_then(Json::as_f64), Some(1.0));
        assert_eq!(value.get("d").and_then(Json::as_f64), Some(-250.0));
        assert_eq!(value.get("c").and_then(Json::as_str), Some("x"));
        let array = value.get("b").and_then(Json::as_array).expect("array");
        assert_eq!(array, [Json::Bool(true), Json::Bool(false), Json::Null]);
    }

    #[test]
    fn decodes_escapes_including_surrogate_pairs() {
        let value = parse(r#""tab\there \"q\" \\ \u00e9 \ud83d\ude00""#).expect("valid string");
        assert_eq!(
            value.as_str(),
            Some("tab\there \"q\" \\ é \u{1F600}"),
            "escapes must decode, including a surrogate pair"
        );
    }
    #[test]
    fn keeps_multi_byte_characters_intact() {
        // 2-, 3- and 4-byte sequences in one string.
        let parsed = parse("\"é—😀\"").expect("valid string");
        assert_eq!(parsed.as_str(), Some("é—😀"));
    }

    #[test]
    fn rejects_the_documents_that_hide_mistakes() {
        // A trailing half-document, a duplicated key, an empty document.
        assert!(parse(r#"{"a": 1} {"b": 2}"#).is_err());
        assert!(parse(r#"{"a": 1, "a": 2}"#).is_err());
        assert!(parse("").is_err());
        assert!(parse("[1, 2").is_err());
        assert!(parse(r#""unterminated"#).is_err());
        assert!(parse(r#""\q""#).is_err());
        assert!(parse("1e").is_err());
    }

    #[test]
    fn rejects_runaway_nesting_instead_of_overflowing() {
        let deep = "[".repeat(MAX_DEPTH + 2);
        let error = parse(&deep).expect_err("must be rejected");
        assert_eq!(error.message, "nesting is too deep");
    }

    #[test]
    fn writer_output_parses_back_to_the_same_value() {
        let original = parse(
            r#"{"n": [3, 4], "h": 0.5, "rho": 2.65, "id": "default", "ok": true, "z": null}"#,
        )
        .expect("valid");
        let text = to_string_pretty(&original);
        let reparsed = parse(&text).expect("writer must emit valid JSON");
        assert_eq!(reparsed, original);
        // Shape: two-space indent, one element per line.
        assert!(text.contains("\n  \"n\": [\n    3,\n    4\n  ]"), "{text}");
    }

    #[test]
    fn numbers_are_written_in_their_f32_shortest_form() {
        // 2.65f32 widened to f64 is 2.6500000953674316; the file must not say so.
        assert_eq!(format_number(2.65f32.into()), "2.65");
        assert_eq!(format_number(2_592_000.0), "2592000");
        assert_eq!(format_number(0.0), "0");
        assert_eq!(format_number(-0.5), "-0.5");
        // A value only f64 can hold is not silently narrowed: it must survive a
        // write-and-parse round trip unchanged.
        let wide = 1.0e-40_f64;
        assert_eq!(
            format_number(wide).parse::<f64>().expect("parses back"),
            wide
        );
    }
}
