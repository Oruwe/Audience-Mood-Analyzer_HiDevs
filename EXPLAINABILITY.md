# Explainability

## Decision Reasoning

Comments are sentiment-scored individually, embedded as vectors, grouped by
k-means over those vectors, and only then summarised — so a cluster exists
because comments landed measurably close together in embedding space, not
because a model decided in one pass that a theme was interesting. Every
surfaced insight carries the verbatim comments it was derived from, validated
character-by-character against the ingested corpus, so a reader can check the
reasoning against the evidence instead of taking the label on trust.

## Data Inputs

The only input is public YouTube comment text and its metadata — comment id,
video id, author display name, like count, timestamp — fetched through the
YouTube Data API with a plain API key and no OAuth grant. Nothing private is
read, no viewer is tracked across channels, and personally identifying
strings and secrets are redacted by deterministic code before any comment is
sent to a model.

## Known Limitations

It reads a bounded sample, not everything: at most 70 comments per video, by
relevance, so a claim about "your audience" is a claim about the comments
YouTube ranks highest, and a brigaded or heavily moderated section will skew
that sample in ways the agent cannot detect. Sarcasm, in-jokes and
community-specific irony are frequently scored at face value, sentiment for
non-English comments is materially weaker than for English, and once the
daily quota is exhausted the agent refuses new analyses outright rather than
degrading to a partial answer.
