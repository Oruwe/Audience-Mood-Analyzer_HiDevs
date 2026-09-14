"""L1 unit tests — ingestion.normalizer.sanitize_comment_text.

Ported from the old run_test.py::verify_normalizer (SPEC §2: "keep, tested,
correct" — moved into pytest per SPEC §2 / §11 L1, run_test.py retired).
"""

from ingestion.normalizer import sanitize_comment_text


def test_utm_stripped_real_params_kept():
    out = sanitize_comment_text(
        "Full write-up: https://example.com/blog/post?utm_source=x&utm_medium=cpc&id=42 🎉"
    )
    assert "https://example.com/blog/post?id=42" in out
    assert "utm_" not in out


def test_email_masked():
    out = sanitize_comment_text("Questions? Email jane.doe+news@sub.example.co.uk")
    assert "[EMAIL]" in out
    assert "@" not in out.replace("[EMAIL]", "")


def test_both_phone_formats_masked():
    out = sanitize_comment_text("Ring support on +1 555 123 4567 or 555-867-5309.")
    assert out.count("[PHONE]") == 2
    assert "555" not in out


def test_whitespace_collapsed_and_blank_runs_capped():
    out = sanitize_comment_text("Too     many    spaces\tand\ttabs\n\n\n\n\nthen more")
    assert "  " not in out
    assert "\t" not in out
    assert "\n\n\n" not in out


def test_leading_trailing_quotes_stripped():
    out = sanitize_comment_text('   "What a great launch!"   ')
    assert out == "What a great launch!"
