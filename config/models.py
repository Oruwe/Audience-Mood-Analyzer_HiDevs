"""SPEC.md §11.1 — the single place any model is named.

No model string may appear anywhere else in the codebase; every module
imports the constants below instead. tests/test_l1_config_models.py enforces
this by grepping the tree for these literal strings outside this file.

This file exists so that skipping the SPEC §11 bake-off costs nothing: the
Stage A/B/C architecture is fixed, only the strings below change. Running
the bake-off later is a two-line edit here, not a rewrite (§11.1).

Bake-off status: NOT RUN. SPEC §13 "Day 0" calls for running it before any
feature work; this project built without it (§11.1's explicitly-sanctioned
path) because doing it properly needs OpenRouter spend and eval-set curation
that don't fit this build's schedule. Track 1 (the encoder) is the one part
of the bake-off later phases *can* run for free against
evals/test_dataset.json (SPEC §11 Track 1) — do that before trusting
STAGE_A_SENTIMENT's accuracy number. Track 2 (Stage B/C schema-validity)
needs a paid OpenRouter key and is still outstanding.
"""

# ---------------------------------------------------------------------------
# Stage A — local encoder, 100% of comments, free, deterministic (SPEC §4.1)
# ---------------------------------------------------------------------------

STAGE_A_SENTIMENT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# Named directly in SPEC.md §4.1 and §11 Track 1 — not a choice made here.
# XLM-R trained on social-media register (emoji, informal text,
# code-switching), which matters for a Hinglish YouTube comment base.
# Validate with the free Track-1 bake-off before shipping: if it lands
# within a few points of an LLM on evals/test_dataset.json, Stage A is
# settled and most of the inference bill is gone for good.

STAGE_A_EMBEDDINGS = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
# SPEC §11.1's own example picks this exact model as the safe starting
# point: 384-dim, the smallest of the three SPEC-approved embedders (e5-base
# is 768, bge-m3 is 1024). "On a free-tier container, a RAM failure hurts
# more than slightly weaker clustering" (§4.1 deployment constraint) — start
# small, measure RAM headroom on day one, upgrade later by changing this one
# line plus a re-embed (§11.1, §9.1 "dimension is your cheapest lever").

# ---------------------------------------------------------------------------
# Stage B / C — OpenRouter, generative (SPEC §4.1, §7, §11)
# ---------------------------------------------------------------------------
#
# Picks below skip the live §11 bake-off (§11.1 explicitly allows this) and
# are instead reasoned from litellm's maintained cost/capability map
# (github.com/BerriAI/litellm — already a repo dependency), pulled
# 2026-09-13, filtered to `openrouter/*` chat models declaring
# `supports_response_schema: true`. openrouter.ai itself is unreachable from
# this build sandbox (network egress policy), so this is the closest
# available substitute for browsing openrouter.ai/models directly — it is
# NOT a substitute for the real §11 L4 schema-validity eval, which measures
# actual per-provider constrained-decoding behavior rather than a declared
# capability flag. Re-run that eval and swap these two strings before this
# product is trusted with a real creator's channel.

STAGE_B_CLASSIFY = "openrouter/deepseek/deepseek-v4-flash"
# Role: batched classification of the Stage-A-flagged subset (~10-20% of
# comments, 40-60 per call) — intent / is_request / is_confusion. Cheapest
# model in the filtered set (~$0.085 / $0.171 per M input/output tokens)
# with a 1,048,576-token context, comfortably oversized for a 60-comment
# batch. DeepSeek's structured-JSON output is well regarded in practice,
# which is the property this stage actually needs (SPEC §11: "pick Stage B
# on schema reliability and cost").

STAGE_C_SYNTHESIS = "openrouter/qwen/qwen3-max"
# Role: one call per cluster (capped at 8) producing an insight block and
# selecting verbatim quotes. At ~8 calls per analysis, price is noise
# ($0.78 / $3.90 per M) so this is picked on capability, not cost (SPEC
# §11: "pick Stage C on quality"). Qwen's flagship non-thinking model:
# strong general reasoning, well-regarded multilingual/code-switching
# handling — relevant given the Hinglish comment base SPEC §4.1 calls out
# for Stage A — and 262,144-token context, well beyond one cluster's worth
# of comments plus schema.
#
# Rejected: OpenRouter ":free" endpoints (e.g. z-ai/glm-5.2:free) for either
# stage. Free tiers carry rate limits that fight the checkpointed batch job
# in SPEC §8, and §11's own selection metric — schema-validity rate under
# batch load — is exactly what a throttled free endpoint is likeliest to
# fail.
