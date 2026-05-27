//! UTF-8 safe string preview helpers.
//!
//! Provides small helpers to take the first N Unicode scalar values (chars)
//! from a string without slicing by byte index which can panic on multibyte
//! characters.

/// Return the first `n` characters of `s` as a `String` (no ellipsis).
pub fn prefix_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Return a preview of `s` up to `n` characters. If `s` is longer than `n`
/// characters, the returned string will include a trailing ellipsis `...`.
pub fn preview(s: &str, n: usize) -> String {
    let mut prefix = prefix_chars(s, n);
    if s.chars().count() > n {
        prefix.push_str("...");
    }
    prefix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_basic_ascii() {
        let s = "hello world";
        assert_eq!(preview(s, 5), "hello...");
        assert_eq!(preview(s, 20), "hello world");
    }

    #[test]
    fn preview_cjk_characters() {
        // Text with multibyte UTF-8 characters.
        let s = "caf\u{e9} r\u{e9}sum\u{e9} \u{20AC}\u{20AC}";
        // Take first 10 characters
        let p = preview(s, 10);
        // Should not panic and should be at most 13 bytes longer due to ellipsis
        assert!(p.chars().count() <= 13);
        // Prefix of chars should match manual char take
        let manual: String = s.chars().take(10).collect();
        if s.chars().count() > 10 {
            assert_eq!(p, format!("{}...", manual));
        } else {
            assert_eq!(p, manual);
        }
    }
}
