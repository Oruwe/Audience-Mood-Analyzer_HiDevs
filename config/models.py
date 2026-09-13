"""SPEC.md §11.1 — the single place any model is named.

No model string may appear anywhere else in the codebase; every module
imports the constants below instead. tests/test_l1_config_models.py enforces
this by grepping the tree for these literal strings outside this file.

This file exists so that skipping the SPEC §11 bake-off costs nothing: the
Stage A/B/C architecture is fixed, only the strings below change. Running
the bake-off later is a two-line edit here, not a rewrite (§11.1).

Bake-off status: NOT RUN for any stage. Track 2 (Stage B/C, and now Stage A
too — see below) needs a paid OpenRouter key and eval-set curation that
don't fit this build's schedule; all three model picks below are reasoned,
not measured, and are flagged for the real §11 L4 schema-validity eval
before this product is trusted with a real creator's channel.

--- Stage A architecture deviation from SPEC.md §4.1 (recorded 2026-09-13) ---
SPEC §4.1 locks Stage A to a local, free, deterministic encoder, for four
explicit reasons: cost (runs on 100% of comments), determinism (testable —
"you cannot unit-test an LLM"), latency/no-quota, and — the one SPEC calls
out as the one people miss — an encoder has no instruction channel, so it
is structurally immune to prompt injection (§10 invariants 1/2 lean on
this for Stage A specifically).

This build's operator explicitly chose to move Stage A to OpenRouter
instead, after being told all four tradeoffs above no longer hold — see
SPEC.md §4.1's amendment note for the full accepted-tradeoff record. In
short: Stage A is no longer free (it now runs an LLM call over every
comment, not just the ~10-20% Stage B sees — cost is the dominant design
constraint below, more so than for Stage B/C), no longer strictly
deterministic, and no longer structurally injection-immune (comment text
now flows into an LLM's context at 100% coverage instead of 0%; SPEC.md's
security-model section is amended accordingly rather than silently left
inconsistent with the code). engine/stage_a.py's batching guard (§4.1b,
originally specified for Stage B) is applied to Stage A's sentiment calls
for exactly this reason — it's no longer just Stage B's problem.
"""

# ---------------------------------------------------------------------------
# Stage A — OpenRouter, 100% of comments (SPEC §4.1, amended 2026-09-13 —
# see the module docstring above and SPEC.md §4.1)
# ---------------------------------------------------------------------------

STAGE_A_SENTIMENT = "openrouter/qwen/qwen3.7-flash"
# Runs on every comment (not a filtered subset like Stage B/C), so cost and
# throughput dominate the pick over raw quality — this is a five-way
# sentiment label, about the easiest task an LLM can be asked to do.
# Cheapest non-free, declared-structured-output model found in the same
# litellm-cost-map survey used for Stage B/C (config/models.py's prior
# revision): ~$0.03 / $0.13 per M input/output tokens, cheaper than
# STAGE_B_CLASSIFY. ":free" tier endpoints were rejected for the same
# reason as Stage B/C: rate limits that fight a checkpointed batch job
# (SPEC §8) and are likeliest to fail exactly the metric SPEC §11 says to
# measure (schema-validity rate under batch load).

STAGE_A_EMBEDDINGS = "openrouter/qwen/qwen3-embedding-0.6b"
EMBEDDING_DIM = 1024
# OpenRouter added a dedicated, OpenAI-compatible /embeddings endpoint
# (confirmed via web search — this postdates litellm's cost map, which has
# zero openrouter/* embedding-mode entries as of the 2026-09-13 pull used
# for Stage B/C, so engine/stage_a.py calls this endpoint directly over
# httpx rather than through litellm). Picked the smallest/cheapest member
# of the Qwen3-Embedding family (~$0.01/M tokens, cheapest embedding option
# found) in the same "start small" spirit as the original local e5-small
# pick — this now matters for cost/latency instead of RAM, but the
# principle is the same: upgrading to the 4B/8B variant later is a
# one-line change plus a re-embed. EMBEDDING_DIM=1024 is the model's
# documented native output size — NOT yet confirmed against a live API
# response (openrouter.ai is unreachable from this build sandbox); verify
# with one real call before trusting downstream clustering math.

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
# — and 262,144-token context, well beyond one cluster's worth of comments
# plus schema.
#
# Rejected: OpenRouter ":free" endpoints (e.g. z-ai/glm-5.2:free) for any
# of the three stages above. Free tiers carry rate limits that fight the
# checkpointed batch job in SPEC §8, and §11's own selection metric —
# schema-validity rate under batch load — is exactly what a throttled free
# endpoint is likeliest to fail.
