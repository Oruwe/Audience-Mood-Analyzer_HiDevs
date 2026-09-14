"""L1 unit test — SPEC §11.1's "single config boundary" rule.

"Create config/models.py ... as the *single* place any model is named. No
model string appears anywhere else in the codebase — that's a testable
rule." This test is that rule.
"""

import re
from pathlib import Path

from config import models

REPO_ROOT = Path(__file__).resolve().parent.parent

_EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", "venv", "env", "node_modules"}
_EXCLUDED_FILES = {
    REPO_ROOT / "config" / "models.py",
    Path(__file__).resolve(),  # this file legitimately imports+names them for comparison
}

_MODEL_STRINGS = [
    models.STAGE_A_SENTIMENT,
    models.STAGE_A_SENTIMENT_FALLBACK,
    models.STAGE_A_EMBEDDINGS,
    models.STAGE_A_EMBEDDINGS_FALLBACK,
    models.STAGE_B_CLASSIFY,
    models.STAGE_B_CLASSIFY_FALLBACK,
    models.STAGE_C_SYNTHESIS,
    models.STAGE_C_SYNTHESIS_FALLBACK,
]


def _repo_python_files():
    for path in REPO_ROOT.rglob("*.py"):
        if path in _EXCLUDED_FILES:
            continue
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def test_model_strings_appear_only_in_config_models():
    offenders: list[str] = []
    for path in _repo_python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for model_string in _MODEL_STRINGS:
            if model_string in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {model_string!r}")
    assert not offenders, (
        "Model ID string(s) found outside config/models.py (SPEC §11.1 single "
        "config boundary rule violated):\n" + "\n".join(offenders)
    )


def test_embedding_dim_matches_stage_a_embedder():
    # If STAGE_A_EMBEDDINGS changes, EMBEDDING_DIM must change with it (SPEC
    # §9.1: "dimension is your cheapest lever" — but only if the two
    # constants stay in sync). Dimensions below are documented/native sizes,
    # not independently confirmed against a live API response — see
    # config/models.py's note on STAGE_A_EMBEDDINGS.
    known_dims = {
        "intfloat/multilingual-e5-small": 384,
        "intfloat/multilingual-e5-base": 768,
        "BAAI/bge-m3": 1024,
        "openrouter/qwen/qwen3-embedding-0.6b": 1024,
        "openrouter/qwen/qwen3-embedding-4b": 2560,
        "openrouter/qwen/qwen3-embedding-8b": 4096,
        "openrouter/openai/text-embedding-3-small": 1536,
        "openrouter/openai/text-embedding-3-large": 3072,
        "openrouter/openai/text-embedding-ada-002": 1536,
    }
    expected = known_dims.get(models.STAGE_A_EMBEDDINGS)
    if expected is not None:
        assert models.EMBEDDING_DIM == expected
    else:
        # Unrecognised embedder swapped in — at minimum the dimension must be
        # a sane positive integer someone deliberately set.
        assert isinstance(models.EMBEDDING_DIM, int) and models.EMBEDDING_DIM > 0


# ---------------------------------------------------------------------------
# The gap the test above leaves open.
#
# `test_model_strings_appear_only_in_config_models` greps for the CURRENT
# values of the constants, so it only enforces the boundary for whatever
# config/models.py happens to name today. A stale slug from a provider this
# project no longer uses sails straight past it -- which is exactly what
# happened: three leftover debug scripts sat in the repo root hardcoding
# `groq/openai/gpt-oss-20b` and `gemini/gemini-3.6-flash` long after both
# providers were dropped. Nothing failed, and anyone running them to check
# their setup would have seen two errors and concluded their keys were bad.
#
# So this one matches the *shape* of a provider slug rather than a list of
# known values, over production code only. Tests legitimately name models --
# placeholder slugs for routing assertions, and the dimension registry above.
# ---------------------------------------------------------------------------

_PROVIDER_SLUG = re.compile(
    r"""["'](?:openrouter|gemini|groq|openai|anthropic|mistralai|deepseek"""
    r"""|google|qwen|meta-llama|cohere|perplexity|xai)/[A-Za-z0-9._:-]+"""
    r"""(?:/[A-Za-z0-9._:-]+)?["']"""
)


def test_no_model_shaped_literal_survives_in_production_code():
    offenders: list[str] = []
    for path in _repo_python_files():
        if "tests" in path.parts:
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            for hit in _PROVIDER_SLUG.findall(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {hit}")
    assert not offenders, (
        "Model-shaped string literal(s) outside config/models.py. Every model "
        "this project names lives there and nowhere else (SPEC §11.1), "
        "including ones no longer in use — a stale slug in a helper script is "
        "worse than a live one, because it fails in a way that looks like the "
        "user's fault:\n" + "\n".join(offenders)
    )
