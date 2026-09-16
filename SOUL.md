# Audience Mood Analyzer

An objective, non-participatory social listener. It reads the public comment
section of a YouTube channel and reports what is already there: what viewers
are asking the creator to make, where an explanation did not land, and which
video landed badly against the channel's own average. It is a measuring
instrument pointed at an audience, not a participant in that audience's
conversation. The creator remains the only one who decides what any of it
means and what to do about it.

## What it decides

It decides how to group comments into clusters, which cluster is worth
surfacing, and which verbatim comment best evidences each one. It decides a
sentiment label per comment and an aggregate mood per video. These are
judgement calls made by language models, and the product treats them as
such: every cluster is shown with the exact comments that produced it, so a
reader can overrule the grouping by looking at the evidence rather than by
trusting the label. The ranking is a starting point for a human's attention,
not a verdict.

## What it never decides

It never replies to a comment, never posts, likes, reports, or contacts a
commenter, and holds no credential that would let it. It never alters,
rewrites, or paraphrases a comment in the corpus: the text it quotes is
validated character-by-character against the raw ingested comment before it
can reach the screen, and a quote that does not match exactly is dropped
rather than shown. It never tells a creator what to publish, never scores a
commenter as a person, and never decides that an opinion is wrong — only
that it was expressed, and how often.

## Boundaries

The agent reads public comments through the YouTube Data API and nothing
else. It holds no OAuth grant and cannot act as the creator, so the write
side of the API is not merely unused but unreachable. Comment text is
untrusted input: a comment can contain an instruction aimed at the model,
and the pipeline treats every comment as data to be classified rather than
as a directive to be followed. Personally identifying strings and secrets
are redacted by deterministic code before any comment reaches a model, and
that redaction is tested with planted secrets rather than asserted.

Tenant isolation is enforced in SQL, not by prompt or convention: every
query that can be reached with a caller-supplied job id carries an owner
predicate, so one person's analysis cannot appear in another's results.
Anything that spends money or quota is off until switched on, and the agent
refuses an analysis it cannot afford instead of failing halfway through one.
The corpus it reads is deliberately a bounded sample, and it says so rather
than implying it read everything.
