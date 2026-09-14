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

--- All four stages moved back to paid models (2026-09-13, same day) ---
That predicted failure mode showed up the same day, measured live, and was
reversed in two steps as evidence came in rather than all at once:

Step 1 (Stage A): a real analysis spent 6+ minutes in Stage A alone,
hitting "upstream_provider_shared_pool" 429s on effectively every batch.
Stage A is the highest-volume stage (100% of comments), so free-tier
throttling costs the most wall-clock time there. The operator approved
paying for Stage A specifically, on the reasoning that Stage B/C's much
lower call volume made the same risk cheaper to carry there.

Step 2 (everything else): that reasoning was wrong, and the logs said so
within minutes. With Stage A fast, the same analysis simply moved its
stall into Stage B, which hit the identical shared-pool 429 on nearly
every batch. Volume was never what made a stage slow — sharing a capacity
pool with every other free user of that model was. All four stages are
now paid, chosen for latency as much as price (~$0.01-0.18 per M tokens
each; a full analysis costs well under a cent), on four *different*
vendors so no single provider's capacity event can stall more than one
stage.

The *_FALLBACK constants were initially left on free models, on the
reasoning that insurance only pays out rarely so its rate limits cost
nothing. That was wrong for one specific reason: insurance is only
insurance if it answers when called, and the same day's logs show the
free pool 429ing on *nearly every* request. A free fallback therefore
routes a failing paid primary straight back into the fire, at exactly the
moment reliability matters most. All three fallbacks are now paid too, on
a different vendor from their own primary AND from each other — so no
single provider event can take out two stages' insurance at once. The
added cost is genuinely ~nil: fallbacks only run when a primary has
already failed.

--- Upgraded off the cheapest tier (2026-09-14) ---
Asked to "use a good model to skip rate limiting". Half of that premise
was already handled and worth recording so it isn't re-litigated: the
429s were `limit_source: upstream_provider_shared_pool`, a *free*-tier
artifact, and paying anything at all had already left that pool. Model
strength buys nothing against a rate limit you are no longer subject to.

The other half was real, and visible in the same logs: repeated "batch of
N returned a length/id mismatch; splitting and retrying" from §4.1b's
guard. Every one of those is a model failing to honour the output
contract, and every one costs a doubled call for that batch — so on this
pipeline, structured-output reliability *is* a latency lever, not merely
a quality preference. All three generative stages moved up a tier for
that reason. A 500-comment analysis now costs roughly $0.03 all-in
(~30 per dollar), against a few tenths of a cent before — bought back in
retries not taken, and in Stage C quotes that survive validation.

--- Why not the local encoder SPEC §4.1 originally specified? (2026-09-13) ---
Asked directly, and worth recording since every failure above is exactly
what §4.1 predicted would happen once Stage A left the encoder. The
encoder is still the better architecture in principle — deterministic,
no quota, injection-immune — but not on this deployment: Render's free
plan gives this service 512 MB RAM and 0.1 CPU, and the running app
already sits at ~440 MB of that. PyTorch alone is ~300-500 MB resident
before any weights load; a base-size sentiment encoder is another ~1.1 GB.
Even ONNX Runtime with int8-quantized MiniLM (~150-200 MB) doesn't fit the
~65 MB of headroom. And 0.1 CPU would make encoder inference over a few
hundred comments slower than the API calls it replaced, since a paid
endpoint runs on the provider's GPU rather than a tenth of a shared core.
So the encoder's "free and fast" inverts to "needs a $25/mo instance
upgrade and is slower" here, versus ~$0.01 per analysis paid. Revisit this
the day the app runs somewhere with >=2 GB RAM and a real core — SPEC
§11.1's whole point is that switching back is a few lines in this file.

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

STAGE_A_SENTIMENT_FALLBACK = "openrouter/google/gemini-2.5-flash-lite"
# $0.100/$0.400 per M. Different vendor (Google) from the primary
# (OpenAI), and a peer of it rather than a downgrade: a fallback that is
# materially worse than its primary silently degrades quality at exactly
# the moment nobody is watching, which is its own kind of outage.

STAGE_A_SENTIMENT = "openrouter/openai/gpt-4.1-nano"
# Runs on every comment (not a filtered subset like Stage B/C), so this is
# the highest-volume call in the whole pipeline — a five-way sentiment
# label, about the easiest task an LLM can be asked to do.
# $0.100/$0.400 per M.
#
# Upgraded from the cheapest-available tier (mistral-nemo, $0.019/$0.030)
# for a reason that is NOT rate limiting -- that was a free-tier
# shared-pool artifact and paying anything at all already fixed it. The
# actual problem the upgrade targets is in the same logs: repeated
# "batch of N returned a length/id mismatch; splitting and retrying"
# from engine/batching.py's §4.1b guard. Each of those is the model
# breaking the output contract, and each one costs a *doubled* call for
# that batch -- so contract adherence is a latency lever, not just a
# quality one. Structured-output reliability is what is being bought
# here; at 5x the old price it is still ~$0.007 for a 500-comment
# analysis, which remains noise next to the retries it avoids.

STAGE_A_EMBEDDINGS_FALLBACK = "openrouter/openai/text-embedding-ada-002"
# Deliberately the SAME vendor as the primary below, breaking this file's
# usual different-vendor-fallback rule -- and that break is required, not
# careless: unlike a chat completion (fixed JSON schema regardless of
# which model answers), an embedding's output width is a property of the
# model itself. A fallback at a different width would corrupt
# engine/insights.py's clustering (KMeans over vectors of inconsistent
# size) rather than merely running on a different model. ada-002 and
# text-embedding-3-small share the 1536 dimension below by construction,
# which is the actual selection constraint here.

STAGE_A_EMBEDDINGS = "openrouter/openai/text-embedding-3-small"
EMBEDDING_DIM = 1536
# Live incident (2026-09-13): the previous pick, qwen/qwen3-embedding-0.6b,
# 404'd with OpenRouter's own "No endpoints found for
# qwen/qwen3-embedding-0.6b" -- caught by harness/preflight.py before a
# real analysis hit it, for $0.00015. That was the SECOND embedding model
# this file has picked that turned out unreliable on OpenRouter (the free
# liquid/lfm-2.5-embedding-350m before it was never verified live either).
# Both were smaller/niche-vendor picks; OpenAI's embeddings are as close
# to commodity infrastructure as this ecosystem has; every serious LLM
# aggregator serves them, and OpenAI has no incentive to deprecate one of
# its most-used endpoints. Preferring that reliability over vendor
# novelty is the point of this specific swap. Cost ~$0.02/M tokens.
# engine/stage_a.py now also gives this stage the same fallback-on-failure
# protection the other three stages have had all along (previously
# embeddings had none at all -- a single 404 killed the whole pipeline
# with no recourse, exactly what just happened).
#
# litellm's cost map still has zero openrouter/* embedding-mode entries at
# any price, for either model above -- this pick and its dimension are
# NOT registry-verified, only harness/preflight.py verifies them, against
# the real live endpoint, before they're trusted with real spend.

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

STAGE_B_CLASSIFY_FALLBACK = "openrouter/deepseek/deepseek-v3.2"
# $0.269/$0.400 per M -- unusually cheap output pricing for its tier, which
# suits Stage B's shape (large batched input, small JSON out). Different
# vendor (DeepSeek) from the primary (OpenAI) and from every other stage's
# fallback, so no single provider event can take out two stages' insurance
# at once.

STAGE_B_CLASSIFY = "openrouter/openai/gpt-4.1-mini"
# $0.400/$1.600 per M. Stage B is the judgement-heaviest of the batched
# stages: is_request vs is_confusion are genuinely ambiguous calls that a
# weak model collapses into "whatever the intent label was" (see the
# prompt in engine/llm_client.py, which now spells out that they are
# independent). It also sees only the Stage-A-flagged subset (~10-20% of
# comments), so a higher unit price applies to a fraction of the volume --
# roughly $0.007 on a 500-comment analysis.
# Role: batched classification of the Stage-A-flagged subset (~10-20% of
# comments, 40-60 per call) — intent / is_request / is_confusion. Paid, for
# the third and last time this file has had to record the same lesson: the
# free predecessor (google/gemma-4-31b-it:free, itself promoted after
# minimax/minimax-m2.7:free was withdrawn from the free tier mid-session)
# hit "upstream_provider_shared_pool" 429s on effectively every batch of a
# real analysis, falling back on each one. The "Stage B is lower volume so
# throttling costs less time there" assumption in this file's Stage A note
# above was measured and found wrong -- once Stage A was fast, Stage B
# became the bottleneck instead. $0.03/$0.13 per M tokens; a "flash"-class
# model, picked for latency as much as price, from litellm's cost map
# filtered to paid `openrouter/*` chat models declaring
# supports_response_schema. Different vendor (Qwen) from Stage A's Mistral,
# so one provider's capacity event can't stall both stages at once.

STAGE_C_SYNTHESIS_FALLBACK = "openrouter/openai/gpt-4.1-mini"
# $0.400/$1.600 per M. A true peer of the primary, deliberately: Stage C
# is the only stage whose output a human reads, and a failure here wastes
# every earlier stage's completed work. Different vendor (OpenAI) from the
# primary (Google).

STAGE_C_SYNTHESIS = "openrouter/google/gemini-2.5-flash"
# $0.300/$2.500 per M, ~8 calls per analysis (~$0.014). SPEC §11 says
# "pick Stage C on quality, price is noise at ~8 calls", and this stage
# has the hardest constraint in the pipeline: every quote it returns is
# checked character-by-character against the real comment corpus
# (schemas.py's _validate_quotes_verbatim), and a single "tidied" quote
# discards the whole insight block. Exact-copy discipline under a long
# instruction is precisely what separates model tiers here.
# Role: one call per cluster (capped at 8) producing an insight block and
# selecting verbatim quotes. Paid, alongside Stage A/B -- at ~8 calls per
# analysis the free tier's throttling risk was the lowest here, but it was
# never zero, and one un-recoverable stall at the very last stage fails a
# report every earlier stage already paid for. SPEC §11 says "pick Stage C
# on quality, price is noise at ~8 calls"; within the paid set that still
# leaves room to prefer a fast one, since this stage is now on the
# critical path of a <40s latency target. $0.065/$0.18 per M tokens, a
# "flash"-class model with a 1.3M-token context (far beyond one cluster
# plus schema), and a third distinct vendor (DeepSeek) after Stage A's
# Mistral and Stage B's Qwen -- no two stages share a provider, so no
# single capacity event can stall more than one.
#
# All four stages are now paid. The free-tier experiment recorded in this
# file's module docstring ran for one session and was reversed stage by
# stage as each one's throttling was measured live rather than predicted.
# The *_FALLBACK constants deliberately stay on free models: they are
# insurance against a genuine outage of a paid primary, not the normal
# path, so their rate limits cost nothing until the day they're the only
# thing still answering.
