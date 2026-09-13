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

--- All four stages moved to OpenRouter ":free" endpoints (2026-09-13) ---
This build's operator explicitly requested every stage — Stage A sentiment,
Stage A embeddings, Stage B, and Stage C — run on a free OpenRouter tier,
reversing the "Rejected: :free endpoints" call this file made earlier the
same day. That earlier rejection's reasoning still holds as a real
tradeoff, not a mistake: free endpoints carry lower, sometimes-changing
rate limits that can collide with SPEC §8's checkpointed batch job, and
§11's own selection metric (schema-validity rate under batch load) is
exactly what a throttled endpoint is likeliest to fail. The operator was
told this and chose free anyway — the priority here is $0 spend over
throughput/reliability margin, which this codebase is already positioned
to absorb better than most: engine/batching.py's split-and-retry guard and
resilience.py's tenacity-backed retry_transient both exist specifically to
turn a transient failure (a 429 included) into a slower success instead of
a crash. If real usage shows free-tier throttling making analyses too slow
or too failure-prone, the fix is a one-line swap back to a paid string
here — same "bake-off is a two-line edit" property this file was already
designed around.

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

STAGE_A_SENTIMENT_FALLBACK = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
# Live incident (2026-09-13): STAGE_A_SENTIMENT alone failed a real analysis
# and the live accuracy benchmark with a genuine, retry-surviving 429 --
# "google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream ...
# limit_source: upstream_provider_shared_pool" -- OpenRouter's free tier
# shares each model's underlying provider capacity across every OpenRouter
# user calling it for free, and Stage A is the highest-volume stage (100%
# of comments, not a filtered subset), so it's the one most likely to hit
# this. resilience.py's existing retry (4 attempts, backoff to 8s) is for
# a momentary blip, not a sustained shared-pool exhaustion, and didn't
# clear it. A different vendor (Nvidia, not Google) was picked as the
# fallback specifically so a Google-side capacity event doesn't take out
# both the primary and the fallback at once. engine/batching.py tries this
# only after the primary has already exhausted its own retries.

STAGE_A_SENTIMENT = "openrouter/google/gemma-4-26b-a4b-it:free"
# Runs on every comment (not a filtered subset like Stage B/C), so this is
# the highest-volume call in the whole pipeline — a five-way sentiment
# label, about the easiest task an LLM can be asked to do. Picked from
# litellm's maintained cost map (github.com/BerriAI/litellm), filtered to
# `openrouter/*` chat models tagged `:free` with `supports_response_schema:
# true` (2026-09-13 pull) — a Mixture-of-Experts model with only ~4B active
# params per token, the closest thing in the free set to matching this
# stage's "cheapest/fastest that can still hit the schema" old criterion.

STAGE_A_EMBEDDINGS = "openrouter/liquid/lfm-2.5-embedding-350m:free"
EMBEDDING_DIM = 1024
# litellm's cost map has zero openrouter/* embedding-mode entries at any
# price (confirmed against a fresh 2026-09-13 pull), so unlike the other
# three stages this pick could not be sourced or cross-checked from that
# registry — it's confirmed only via openrouter.ai's own model page
# (openrouter.ai/liquid/lfm-2.5-embedding-350m:free; openrouter.ai is
# unreachable from this build sandbox, so this is a web-search result, not
# a call this build has made itself). Chosen because it's the one free
# OpenRouter embedding model that documents 1,024-dimension output,
# matching EMBEDDING_DIM without a re-derivation — but that dimension, and
# the model's existence/behavior at all, is UNVERIFIED against a live API
# response from this build. The model page also documents a 512-token
# input cap per text, well under this stage's per-comment inputs in
# practice but unenforced here — a comment longer than that may be
# silently truncated by the provider rather than rejected; SPEC §11's
# real eval pass should check this before this product is trusted with a
# real creator's channel. The previously-configured paid alternative
# (qwen/qwen3-embedding-0.6b, ~$0.01/M tokens) was already negligible cost
# and is not the reason this changed — it changed because the operator
# asked for $0 spend on every stage, embeddings included.

# ---------------------------------------------------------------------------
# Stage B / C — OpenRouter, generative (SPEC §4.1, §7, §11)
# ---------------------------------------------------------------------------
#
# Picks below skip the live §11 bake-off (§11.1 explicitly allows this) and
# are instead reasoned from litellm's maintained cost/capability map
# (github.com/BerriAI/litellm — already a repo dependency), pulled
# 2026-09-13, filtered to `openrouter/*` chat models tagged `:free` and
# declaring `supports_response_schema: true`. openrouter.ai itself is
# unreachable from this build sandbox (network egress policy), so this is
# the closest available substitute for browsing openrouter.ai/models
# directly — it is NOT a substitute for the real §11 L4 schema-validity
# eval, which measures actual per-provider constrained-decoding behavior
# (and, for the free tier specifically, throughput under rate limits)
# rather than a declared capability flag. Re-run that eval and swap these
# strings before this product is trusted with a real creator's channel.

STAGE_B_CLASSIFY_FALLBACK = "openrouter/google/gemma-4-31b-it:free"
# Same shared-pool-exhaustion risk as STAGE_A_SENTIMENT_FALLBACK above,
# lower-probability here only because Stage B sees a filtered subset
# (~10-20% of comments) rather than 100% of them. Different vendor
# (Google) from the primary (MiniMax) for the same reason.

STAGE_B_CLASSIFY = "openrouter/minimax/minimax-m2.7:free"
# Role: batched classification of the Stage-A-flagged subset (~10-20% of
# comments, 40-60 per call) — intent / is_request / is_confusion. Picked
# from the free+schema-capable set on context headroom (196,608 tokens,
# comfortably oversized for a 60-comment batch) and general standing as a
# mid-size, instruction-tuned model — the free-tier analogue of the old
# "pick Stage B on schema reliability and cost" criterion (SPEC §11), with
# cost now fixed at $0 across the whole free set and reliability the only
# remaining axis to differentiate on.

STAGE_C_SYNTHESIS_FALLBACK = "openrouter/minimax/minimax-m3:free"
# Same shared-pool-exhaustion risk, lowest-probability of the three
# generative stages (only ~8 calls total) but still worth covering: a
# single unfallback-able failure here fails the whole report at the very
# last stage, after every other stage already succeeded. Different vendor
# (MiniMax) from the primary (Z-AI/GLM).

STAGE_C_SYNTHESIS = "openrouter/z-ai/glm-5.2:free"
# Role: one call per cluster (capped at 8) producing an insight block and
# selecting verbatim quotes. At ~8 calls per analysis this was previously
# picked on capability alone since price was noise (SPEC §11: "pick Stage C
# on quality") — with cost now fixed at $0 across the whole free set,
# that same "pick on capability" logic points at GLM's flagship free
# entry: it's the frontier-class model in the free+schema-capable set
# (openrouter/openrouter/auto and openrouter/openrouter/free were also
# available but were rejected here — they're OpenRouter's own dynamic
# meta-routers, not a pinned model, which would silently reintroduce the
# non-determinism this project already gave up once for Stage A; a fixed
# string keeps this stage swappable-and-testable the same way as the
# other three). 256,000-token context, well beyond one cluster's worth of
# comments plus schema.
#
# All three generative stages above (and Stage A embeddings, sourced
# separately — see its own comment) are free-tier, per the operator's
# explicit "use a free model" request covering every stage. See this
# file's module docstring for the accepted-tradeoff record on that
# decision.
