# Audience Mood Analyzer — Creator Edition

**Build spec. Hand this to Claude Code one section at a time, not all at once.**

---

## 1. What this is now

**One sentence:** A creator pastes a YouTube channel or video URL and gets back what their
audience is actually asking for, where their explanations failed, and which video landed badly —
each claim backed by real comments.

| | |
|---|---|
| **User** | YouTube creators with 1k–100k subs. Reachable by DM. Underserved by enterprise tools. |
| **Input** | A channel URL or video URL. Nothing else. No signup for the first run. |
| **Output** | Three insight blocks, each citing verbatim comments. |
| **Why this platform** | YouTube's Data API is free within quota. X and Reddit are paid in 2026 — they kill a student-run product before it has a user. |

**What it is not:** a real-time brand-monitoring radar. That was V2's framing and it's why the
architecture had three processes. A creator doesn't need a live firehose. They need a good read
on comments they already have.

---

## 2. Keep / cut / build

### Keep (this code is good — do not rewrite it)

- `ingestion/normalizer.py` — PII masking, UTM stripping, whitespace collapse. Tested, correct.
- `ingestion/dedup.py` — fingerprint + LRU/TTL. Repurpose: kill bot spam and copypasta within a channel.
- `engine/llm_client.py` — tiered Gemini → Groq routing with Pydantic validation. Keep the routing and the contract enforcement; rewrite only the call shape (see §4).
- `schemas.py` — keep, extend.
- `evals/benchmark.py` + `evals/test_dataset.json` — keep. **This time commit the output.**
- `run_test.py` — keep the checks, move them into `tests/` as pytest (see §6).

### Cut

- `ingestion/bluesky_stream.py` — wrong platform.
- `engine/anomaly_detector.py` — crisis radar. Wrong framing. Delete or park on a branch.
- `pipeline.py` (3-process firehose runner + token bucket) — replaced by a request-scoped job.
- `server.py` — the FastAPI hop existed to keep DB credentials out of the console. For one
  Streamlit app it is ceremony, and it is the single biggest reason nothing ran on clone.
- `storage/db.py` DuckDB-on-local-disk + migration machinery — Streamlit Cloud and HF Spaces have
  ephemeral disk. A local `.duckdb` file vanishes on every redeploy.
- `recover_legacy.py` — gone with the above.

### Build new

- `ingestion/youtube.py` — channel → video list → comment pagination, with quota accounting.
- `engine/insights.py` — **the differentiated layer. This is the product.**
- `app.py` — single Streamlit entrypoint. No API hop.

---

## 3. The insight contract

This is the part V2 skipped, and it is the only part that isn't a commodity. Every competitor
can classify sentiment. Almost none tell a creator what to *do*.

Three blocks. **Every single one must cite verbatim comments.** An LLM summary with no receipts
is what makes these tools feel fake, and users can smell it immediately.

**Block 1 — Requests.** What the audience is asking you to make.
```
theme: str                  # "Wants a Docker follow-up"
mention_count: int
quotes: list[str]           # 2-3 verbatim, unedited
suggested_title: str        # a video title that answers it
```

**Block 2 — Confusion points.** Where your explanation didn't land.
```
sticking_point: str         # "Lost people at the env var setup"
mention_count: int
quotes: list[str]
timestamp_hint: str | None  # if commenters cited a timestamp
```

**Block 3 — Mood by video.** Which video underperformed emotionally, and why.
```
video_title: str
sentiment_score: float
delta_vs_channel_avg: float
top_negative_driver: str
quotes: list[str]
```

Block 2 is the sleeper. For any tutorial or explainer channel it's a direct
next-video generator, and no mainstream tool produces it.

---

## 4. Critical engineering decisions

### 4.1 Three-stage cascade. Do not send every comment to an LLM.

V2 made **one LLM call per comment**. Correct for a 1-comment-every-2-seconds firehose.
Catastrophic for a channel backfill: 5,000 comments = 5,000 calls. That blows every free tier,
takes hours, and costs real money.

The fix is not just batching — it's putting an encoder in front of the LLM.

**Stage A — encoder-only, local, free, deterministic. 100% of comments.**
- Sentiment: `cardiffnlp/twitter-xlm-roberta-base-sentiment`. XLM-R trained on social-media
  register — emoji, informal text, code-switching. Matters for a Hinglish comment base.
- Embeddings: `BAAI/bge-m3` or `intfloat/multilingual-e5-base`, run locally.
- This **deletes V2's hash-vector embedding fallback**, which silently made clustering
  meaningless whenever it fired. You now always have real vectors.

**Stage B — generative, batched, on the filtered subset only (~10–20% of comments).**
- Only what Stage A flags: strong negative, high-confidence, or inside a dense cluster.
- 40–60 comments per call, returning a JSON array of `intent`, `is_request`, `is_confusion`.
- These labels are genuinely beyond an encoder without labelled training data. The LLM earns
  its place here and nowhere else in the per-comment path.

**Stage C — generative, synthesis. ~8 calls.**
- One call per cluster, capped at 8, producing the insight block and selecting verbatim quotes.

Net: 5,000 comments → **~20 LLM calls**, down from 5,000. Batching alone would have got you to
~110; the encoder pre-filter takes another 5x out.

Four reasons this is the right shape, and the fourth is the one people miss:

1. **Cost** — Stage A is free and runs on CPU.
2. **Determinism** — same input, same output. Your L1 tests can assert exact values. You cannot
   unit-test an LLM.
3. **Latency** — milliseconds, no rate limit, no quota, no network.
4. **An encoder is structurally immune to prompt injection.** It has no instruction channel. A
   comment reading "ignore previous instructions" is just tokens to embed. Invariants 1 and 2
   from §10 come free from the architecture rather than from defending against anything.

**Deployment constraint — check this before committing.** A base-size XLM-R plus bge-m3 is a lot
of RAM for a free Streamlit Cloud container. Measure it. If it doesn't fit: int8-quantised ONNX
versions (roughly 4x smaller, faster on CPU), or drop to a MiniLM-class embedding model, or move
the host to Render / HF Spaces. Find out on day one, not on deploy day.

> **Amendment, 2026-09-13 — Stage A moved to OpenRouter.** This build's operator was told all
> four reasons above in full before deciding, and chose to route Stage A through OpenRouter
> (`config/models.py::STAGE_A_SENTIMENT`, `STAGE_A_EMBEDDINGS`) instead of a local encoder.
> Accepted tradeoffs, explicitly:
>
> 1. **Cost** no longer holds — Stage A now runs a paid LLM call over 100% of comments, not the
>    ~10–20% Stage B sees. This is the dominant cost driver in the whole pipeline now, more so
>    than Stage B or C.
> 2. **Determinism** no longer holds structurally — an LLM call is not guaranteed to return the
>    same label for the same input twice. `engine/stage_a.py`'s tests assert *shape and
>    behavior* (batching, the §4.1b guard, error handling) against a stubbed API, not that a
>    real call is reproducible.
> 3. **Latency/quota** no longer holds — Stage A is now on the network, subject to OpenRouter
>    rate limits and to the same "zero API keys" demo-fixture requirement (§6) as Stage B/C:
>    the committed demo channel's cached analysis is what makes the 60-second zero-key demo
>    work, not Stage A running offline.
> 4. **Prompt-injection immunity is gone for Stage A.** This is the one that changes the
>    security model, not just the cost model: comment text now flows into an LLM's context at
>    100% coverage instead of 0%. §10's invariants 1 and 2 no longer "come free from the
>    architecture" for Stage A the way this section originally argued — they now depend on the
>    same schema-constrained-output defense Stage B/C always needed (Pydantic `Literal`/`Enum`
>    fields, no free strings), which `engine/stage_a.py` applies via `StageASentimentBatch`.
>    There is no local, non-LLM stage left in this pipeline that comment text bypasses.
>
> The §4.1b batching guard below — written with Stage B in mind — now applies to Stage A's
> sentiment calls too, for exactly this reason, and `engine/stage_a.py` implements it there.
> §7's stack table and §10's invariant table are updated to match.

### 4.1b Batching guard

**Required guard:** validate that the returned array length equals the input batch length. Batched
structured output silently drops or merges items under load. On mismatch, split the batch in half
and retry. Do not skip this — it corrupts results in a way that looks like working output.

### 4.2 Cache by video, not by request

Key on `(video_id, comment_count)`. If the comment count hasn't moved, serve the cached analysis.
This makes repeat visits free and is what lets you survive going viral on a free tier.

### 4.3 Persistence

Not local DuckDB — ephemeral disk kills it. Use Neon or Supabase free-tier Postgres: one
connection string, results survive redeploys. Results surviving redeploys is what makes the demo
credible.

### 4.4 Verify quotas before designing around them

Two numbers to look up and write into this spec **before writing code**:

- YouTube Data API daily quota + the real cost of `commentThreads.list` pagination.
  Compute your quota cost per channel analysis and refuse cleanly when it would exceed budget —
  never half-fail.
- Gemini's current free-tier RPM/RPD. The free tier was cut substantially during 2026, so do not
  design around remembered numbers. Read the docs.

If a channel analysis would exceed quota, say so in the UI up front. A clear refusal reads as
competence; a spinner that dies reads as broken.

> **Filled in 2026-09-13** (developers.google.com is unreachable from the build sandbox's network
> egress policy; numbers below are cross-checked across multiple independent current sources —
> re-verify against the official docs before a production deploy).
>
> **YouTube Data API v3.** Default quota is **10,000 units/day**, per Google Cloud *project* (not
> per key — every API key in one project draws the same pool), resets at midnight Pacific Time,
> no rollover. `search.list` is the expensive outlier at 100 units/call; every list method this
> product actually needs is a flat **1 unit per call regardless of `part`/`maxResults`**:
>
> | Call | Cost | Max page size | Used for |
> |---|---|---|---|
> | `channels.list` (`forHandle` or id, `part=contentDetails,statistics`) | 1 | — | resolve channel → uploads playlist id + total video count, in one call |
> | `playlistItems.list` (uploads playlist) | 1 | 50 | enumerate a channel's video IDs |
> | `videos.list` (batched, up to 50 ids/call) | 1 | 50 ids/call | per-video `commentCount` + title, for the pre-flight estimate |
> | `commentThreads.list` | 1 | 100 | the actual comment pull, per page |
>
> Quota cost per channel analysis ≈
> `1 + 2·⌈video_count/50⌉ + Σ⌈comment_count_i/100⌉` over the videos analyzed. `ingestion/youtube.py`
> computes the real right-hand sum from actual per-video `commentCount` (via the batched
> `videos.list` pre-flight, itself only `2·⌈video_count/50⌉` units) and refuses **before** spending
> anything on the comment pull if that would exceed the configured daily budget. In practice the
> binding constraint is many users sharing one deployed project's 10,000-unit pool, not any single
> analysis — see `ingestion/youtube.py`'s in-memory quota ledger.
>
> **Gemini free tier.** Current figures (Gemini 3 Flash, free tier, per project): **~10 RPM,
> ~1,500 RPD, ~250K TPM** — consistent with the spec's warning that the free tier was cut
> substantially during 2026 versus earlier, more generous numbers. **Moot for this build**: §7's
> locked stack routes all generative inference through OpenRouter
> (`config/models.py::STAGE_B_CLASSIFY` / `STAGE_C_SYNTHESIS`), not Gemini directly. The kept
> `engine/llm_client.py` Gemini→Groq chain is the *shape* being preserved (tiered failover +
> Pydantic validation), not the providers — Phase 5 rewrites the actual chain onto the OpenRouter
> picks above.

---

## 5. Architecture

```
Streamlit app.py
   │
   ├─ ingestion/youtube.py ──▶ normalizer ──▶ dedup
   │
   ├─ engine/llm_client.py  ──▶ batched classify (map)
   │
   ├─ engine/insights.py    ──▶ cluster + synthesize (reduce)
   │
   └─ Postgres (Neon) ◀── cache by (video_id, comment_count)
```

One process. One command. That's the whole point.

---

## 6. Definition of done

This section exists because V2 was better engineered than this will be and still scored badly.
**None of the above matters if these aren't true.**

- [ ] `git clone && pip install -r requirements.txt && streamlit run app.py` produces a **working
      app with a populated demo channel and zero API keys.** Commit a cached real analysis as the
      demo fixture. A grader, a recruiter, or a creator must see output in under 60 seconds.
- [ ] Public deployed URL, on **line one** of the README.
- [ ] README opens with: the problem, who it's for, one screenshot, the accuracy number.
      Architecture goes at the bottom, not the top.
- [ ] `data/eval_metrics.json` **committed** — remove `data/` from `.gitignore` or carve it out.
      The accuracy number is your strongest evidence and last time it was invisible.
- [ ] `tests/` directory, real pytest, CI green with a badge.
- [ ] `examples/` — three real channel analyses committed as markdown. This is what makes the
      project credible to someone who never runs it.
- [ ] 90-second demo video linked in the README.

---

## 7. Stack (locked)

| Layer | Choice | Note |
|---|---|---|
| Classification + embeddings | ~~Local encoder-only models~~ **OpenRouter (§4.1 amendment, 2026-09-13)** | Was free/deterministic/injection-immune; now Stage A is a third OpenRouter consumer alongside B/C — see §4.1's amendment note for the accepted tradeoffs |
| Generative inference | OpenRouter, open models | Three picks now (Stage A/B/C, not just B/C): cheap for A (runs on 100% of comments), cheap+schema-reliable for B, strong for C. Chosen by bake-off (§11). |
| Orchestration | Plain async Python | No agent framework. There are no agents. |
| Durable state | Postgres (Neon) | Anything you must not lose |
| Shared/ephemeral state | **None in v1** | Redis arrives with the worker process, not before |
| Vector storage | **None in v1**, then pgvector on the same Postgres | No separate vector DB. See §9.1 |
| Auth | Streamlit native OIDC (`st.login`) + Google | Same Google account owns the channel. One flow. |
| Observability | Langfuse + OTel | Already in repo |
| UI | Streamlit | Background job + `st.status`, never a blocking call |

Rejected: Lyzr (no agents to orchestrate), a guard LLM on the hot path (see §12),
Redis in v1 (single process, nothing to share).

## 8. Orchestration

`analyze_channel(channel_id) -> AnalysisResult` — a plain async function composing the six steps.

- **Concurrency:** `asyncio.Semaphore` over batch calls + the token bucket already written in V2's
  `pipeline.py`. Reuse it.
- **Retries:** `tenacity`, exponential backoff + jitter, on YouTube and OpenRouter calls.
- **Background execution:** `threading.Thread` + a job-status row in Postgres.
- **Checkpointing (required):** write each completed batch to Postgres as it finishes. Streamlit
  Community Cloud sleeps inactive apps and the thread dies with the container. On start, load
  completed batches and skip them — the job function must be idempotent. Without this, a crash at
  comment 4,200 of 5,000 burns the quota twice.

Graduate to Redis + RQ with a worker on Render when analyses exceed ~10 min or users are
concurrent. Temporal if you ever need genuinely durable long-running workflows. Both are week-8
problems.

## 9. State

> **Postgres for anything you must not lose. Redis for anything shared across processes that you
> can afford to lose.**

| State | Where | Why |
|---|---|---|
| Analysis results cache, keyed `(video_id, comment_count)` | Postgres | Must survive redeploys. Redis evicts. |
| Dedup within one analysis | Python `set` | Scoped to one job in one process |
| Follow-up chat history | Postgres | Small, no latency requirement |
| Quota / rate counters | In-memory v1 → Redis at multi-process | Atomic counters with TTL is the one Redis-shaped job |
| OAuth refresh tokens | Postgres, **encrypted at rest** | See §12 |
| Comment embeddings (clustering) | In-memory numpy, discarded after the job | Ephemeral. No storage, no DB. |
| Comment embeddings (question box) | **pgvector on the same Neon Postgres** | No separate vector DB. See below. |

### 9.1 Vectors — no vector database

Vectors appear in exactly two places, and only one of them needs persistence.

**Clustering inside an analysis** is ephemeral: embed the flagged subset, KMeans it, use the
centroids, throw the vectors away. That's numpy. It never touches a database.

**The §12 question box** is the only real retrieval need, and it's small. 5,000 comments at 1024
dims is ~20 MB; a large channel at 50,000 comments is ~200 MB. Brute-force cosine over a *single
channel's* vectors runs in milliseconds. You are not near needing an index, let alone a service.

So: **pgvector on the Postgres you already have.** Three reasons:

1. Zero new infrastructure, zero new credentials to secure.
2. Metadata filtering in the same query — `WHERE user_id = $1 AND channel_id = $2` alongside the
   vector search. A separate store means maintaining tenant filtering in two systems and keeping
   them in sync, which is precisely how cross-tenant leaks happen (§10 invariant 5).
3. **Deletion.** A separate vector store is a second copy of user data with its own deletion path.
   With pgvector, dropping an account takes its vectors in the same transaction. With an external
   store you have to remember — and account deletion is a Google OAuth verification requirement,
   not a nice-to-have.

**Ship v1 with no vector storage at all.** The three insight blocks need clustering, not retrieval.
Add pgvector when you build the question box, not before.

**Change only if** you reach millions of vectors with high QPS — and even then the first move is an
HNSW index in pgvector, not a new service. Realistically this product never gets there.

**Dimension is your cheapest lever** if storage or latency ever bites: bge-m3 is 1024-dim,
multilingual-e5-base is 768, MiniLM-class is 384. Measure it during the §11 embedder bake-off,
since you're benchmarking those models anyway.

## 10. Security model

Assets, in severity order: **user OAuth refresh tokens** (grant access to a creator's private
analytics — leaking these ends the product) → **API keys** (a leaked OpenRouter key gets drained)
→ **user analysis data** → **cross-tenant isolation**.

The design principle: **make escalation structurally impossible rather than trying to detect it.**
A guard model watching for injection is weaker and more expensive than an architecture where a
successful injection can't do anything.

Seven invariants. Each one is enforced by construction and proven by a test.

| # | Invariant | Enforced by | Proven by |
|---|---|---|---|
| 1 | The LLM never receives a secret, token, or another user's data | Prompt builder takes only `(comments, config)` — no DB handle, no session object | Test asserts no value from the secrets store appears in any built prompt |
| 2 | The LLM has no tools; output is a constrained schema | Pydantic with `Literal` types, not free strings | Schema test + fuzz the classifier with injection payloads, assert output stays in-enum |
| 3 | Every quoted comment is real | Verbatim match against source corpus | Hard test. Fails the build. |
| 4 | No HTML/script injection via comment text | `unsafe_allow_html` banned repo-wide | `grep` step in CI, fails on any match |
| 5 | Tenant isolation | Every query scoped by `user_id`, parameterized | Integration test: user A's session cannot read user B's row |
| 6 | Least privilege + encrypted tokens | `yt-analytics.readonly` scope only; Fernet at rest | Test asserts the stored token is not plaintext |
| 7 | Spend cannot run away | Hard credit cap at OpenRouter + per-user rate limit | Load test hits the limiter and gets refused |

**On the AI leaking secrets:** it cannot leak what it never receives. Invariant 1 is the entire
defense and it is stronger than any guard model. Corollary: **treat your system prompt as public.**
Design as though it will be printed on the internet. If a leaked prompt hurts you, the architecture
is wrong — not the prompt.

**Where a guard model is actually worth it:** the synthesis/reduce step only (~8 calls per
analysis), where many untrusted comments get fused into a claim the creator will act on. Never on
the per-comment hot path.

> **Amendment, 2026-09-13 — Stage A is now in scope for invariants 1 and 2.** §4.1's amendment
> moved Stage A onto OpenRouter, so "an encoder has no instruction channel" no longer exempts it
> — comment text now reaches an LLM at 100% coverage, on the per-comment hot path, which is
> exactly the case the paragraph above says never to put a guard model on. The compensating
> control is invariant 2 itself: `StageASentimentBatch` constrains Stage A's output to an
> `Enum` sentiment field and a bounded `float` confidence, same as Stage B/C. A successful
> injection against Stage A can, at most, talk the model into mislabeling *that one comment's*
> sentiment — it still cannot produce a value outside the enum, call a tool, or reach anything
> invariant 1 doesn't already keep out of the prompt. No guard model added here; the schema is
> the containment, per the design principle above, not an exception to it.

**Two more, easy now and painful to retrofit:** scrub PII *before* the Langfuse call, not after
(a token or email interpolated into a traced prompt lives in Langfuse forever), and build the
account-delete path on day one — you need it for Google verification anyway.

## 11. Test harness

Four layers. Layers 1–3 are deterministic tests. Layer 4 is evals — scored, drifting, not pass/fail.
Do not conflate them.

**L1 — Unit, no network.** Normalizer, fingerprinting, quota arithmetic, batch splitting, divergence
quadrant logic. Port V2's `run_test.py` checks. Sub-2s, every commit.

**L2 — Contract tests, stubbed LLM.** The best thing in the V2 repo — keep the approach. Test the
failure modes that actually break production: malformed response, array-length mismatch triggering
split-and-retry, provider failover, 429 backoff, quota refusal. **Security invariants 1, 2, 4, 5, 6
live here.** Free, deterministic, CI-safe.

**L3 — Integration, recorded.** Capture real YouTube and OpenRouter responses once as cassettes
(`vcrpy` / `pytest-recording`), replay in CI, re-record deliberately. Catches API shape changes
without per-run cost or keys in GitHub Actions.

**L4 — Evals.**
- *Classification* — `evals/test_dataset.json` → accuracy, per-class F1, confusion matrix.
  Gate: no prompt or model change merges on a >2pt accuracy drop.
- *Schema validity* — % of batches returning a valid array of correct length, per model.
  **This is how you pick the OpenRouter model.** Almost nobody measures it.
- *Insight quality* — golden set of ~10 channels with human-written expected top-3 requests,
  scored by LLM-as-judge.

**Task zero, before any feature work:** the bake-off, in two tracks against
`evals/test_dataset.json`.

*Track 1 — the encoder.* Run `twitter-xlm-roberta-base-sentiment` over the eval set. Accuracy,
per-class F1, and specifically how it does on the `mixed` and sarcasm rows. Free to run, so this
costs only time. If it lands within a few points of an LLM, Stage A is settled and you've removed
most of your inference bill permanently.

*Track 2 — the generative models.* Three OpenRouter candidates from the current open leaders
(Qwen, DeepSeek, GLM, Llama families), filtered to those declaring structured-output support.
Batch size 50. Measure **schema-validity rate** — % of batches returning a valid array of the
correct length — alongside accuracy. Note that on OpenRouter the *provider* serving a model, not
just the model, determines whether constrained decoding is actually enforced; check both.

Pick Stage B on schema reliability and cost. Pick Stage C on quality — at ~8 calls per analysis a
bigger model costs you almost nothing there.

### 11.1 Building without the bake-off

The bake-off is skippable, and skipping it costs nothing **provided every model ID lives behind one
config boundary.** The architecture does not change — Stage A/B/C are fixed; only the model strings
differ. Done right, running the bake-off later is a config edit, not a rewrite.

Create `config/models.py` (or `models.toml`) as the *single* place any model is named. No model
string appears anywhere else in the codebase — that's a testable rule.

```python
STAGE_A_SENTIMENT  = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
STAGE_A_EMBEDDINGS = "intfloat/multilingual-e5-small"   # 384-dim; upgrade to -base or bge-m3
                                                        # once RAM headroom is measured
STAGE_B_CLASSIFY   = "<openrouter model, structured-output support required>"
STAGE_C_SYNTHESIS  = "<openrouter model, stronger; only ~8 calls per analysis>"
EMBEDDING_DIM      = 384    # must match STAGE_A_EMBEDDINGS
```

Start small on embeddings. On a free-tier container, a RAM failure hurts more than slightly
weaker clustering, and moving up later is one line plus a re-embed.

For Stage B and C, pick from OpenRouter's current catalogue filtered to models declaring
structured-output support, and record *why* in a comment next to each. Then run the bake-off when
convenient and change two strings.

**CI:** L1–L3 every push, no API keys required. L4 on manual trigger and pre-release, results
written to `data/eval_metrics.json` — **committed**. Langfuse closes the loop in production: sample
real traces, score on the same rubric, watch for drift.

### 11.2 In-app evaluation-criteria panels (added 2026-09-13)

`evals/benchmark.py`'s L4 classification eval (accuracy, per-class F1, confusion matrix over
`evals/test_dataset.json`) previously only ran offline (`python -m evals.benchmark`, writing
`data/eval_metrics.json`). It's now also reachable from inside the running app itself: app.py's
"📊 Model accuracy benchmark (live)" panel runs the exact same function
(`run_benchmark(persist_to_file=False)`) against whatever model this deployment is actually
configured with, using its real `OPENROUTER_API_KEY` — a live, on-demand number, not a claimed one
— and persists the result to Postgres (`storage.postgres.model_eval_runs`) instead of a file, so it
survives a restart the same way SPEC §8 job state already does.

Two more panels round out the same "measure it, don't just claim it" principle for the parts of
this product a demo/rubric reviewer can't otherwise see:
- **Mood distribution + per-stage timing charts** (app.py's `_render_insights`) — a real chart of
  the actual sentiment counts and per-stage wall-clock breakdown from the *current* analysis job,
  not just the three text-based insight blocks §3 specifies. Per-stage timing is derived from each
  stage's own checkpoint timestamps (`get_stage_checkpoint_summary`) rather than a dedicated
  profiler — SPEC §8 only asks for a checkpoint per unit of work.
- **Honest throughput, not a "real-time" claim** — `efficiency_summary` reports measured
  comments/sec end-to-end for a completed job. This pipeline is a checkpointed background job
  (§8), not a low-latency streaming system; reporting a real number here is a deliberate choice not
  to overstate what SPEC §8's architecture actually is.

## 12. Flexibility

Do not build a settings page. Configurability is where early products die — every option is a
combinatorial test burden and almost nobody touches it.

- **Fixed and opinionated:** the three insight blocks from §3. Always shown. No toggles.
- **One escape hatch:** a free-text question box over the already-analyzed corpus ("ask anything
  about these comments"). Cheap — no re-analysis, just retrieval over stored results. And the
  questions people type become your roadmap: that box is your cheapest user research.
- **Insight definitions are data, not code.** Each block is a versioned `(prompt, schema)` config
  in git or Langfuse prompt management. Adding a fourth block for gaming creators is a config
  change, not a deploy.

The axis of flexibility is **extensible by you, not configurable by them.**

## 13. Sequence

**Day 0** — The §11 model bake-off. Nothing else starts until a model is chosen on data.

**Days 1–3** — Execute the cuts in §2. Build `ingestion/youtube.py`. Get batched classification
working locally against one real channel (use your own, or a friend's). Success = a JSON file of
labelled comments for a real channel.

**Days 4–5** — Build `engine/insights.py`. All three blocks, with evidence citation enforced in
the schema. Success = you read the output for a channel you know well and it tells you something
true that you didn't already know. If it doesn't, the product doesn't work yet — fix it before
deploying.

**Day 6** — Deploy. README. Demo fixture. Commit examples. Everything in §6.

**Week 2** — Ten creators, by DM. Watch them use it. Fix what breaks. Do not add features.

**Week 3** — Write the arc publicly.

---

## 14. How to drive Claude Code with this

Give it this spec as context, then **one section at a time**. Specifically:

1. "Read SPEC.md. Run the §11 bake-off. Report accuracy and schema-validity per model. Change
   nothing else."
2. "Execute §2 cuts only. Show me the diff."
3. "Set up the §11 L1 + L2 harness against the surviving code, including the §10 invariant tests
   for 1, 2, 4. Red is fine — I want the scaffolding."
4. "Build `ingestion/youtube.py` per §1 and §4.4. Pagination, quota accounting, refuse cleanly
   over budget. Tests first."
5. "Rewrite `engine/llm_client.py` for the §4.1 batched map stage. Include the array-length guard."
6. "Build `engine/insights.py` per §3. Evidence citation enforced in the Pydantic schema, not the
   prompt. Wire up the §10 invariant-3 verbatim check as a hard test."
7. "Build `app.py` per §5 and §8 — background thread, checkpointing, `st.status`. No FastAPI."
8. "Add §10 invariants 5, 6, 7 with their tests."
9. "Work the §6 checklist until every box is true."

Handing an agent the whole spec and saying "build it" is how you get back something that runs on
its machine and nowhere else. That's the failure mode you're recovering from.
