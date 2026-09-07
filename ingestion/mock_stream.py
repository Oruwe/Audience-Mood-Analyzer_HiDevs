"""Mock ingestion stream: async generator simulating multi-platform comments."""

import asyncio
import random
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from schemas import RawComment

# (platform, author, text) — covers praise, sarcasm, acute frustration,
# constructive feedback, pricing skepticism, and neutral inquiries.
COMMENT_POOL: list[tuple[str, str, str]] = [
    # --- Praise ---
    ("twitter", "@maya_builds", "Just shipped our launch with this tool and the analytics dashboard is gorgeous. 10/10, recommending it to every founder I know."),
    ("youtube", "DevWithLena", "This is the only tutorial that actually explained the setup clearly. Instant subscribe!"),
    ("reddit", "u/ops_oliver", "Shoutout to the dev team — the bug I reported Friday was fixed by Monday morning. Rare and appreciated."),
    ("bluesky", "pixelpioneer.bsky.social", "The open-source community around this project is genuinely lovely. Maintainers reply within hours."),
    # --- Sarcasm ---
    ("twitter", "@cynical_sam", "Oh great, another 'AI-powered' rebrand. Because what this product really needed was a chatbot nobody asked for 🙄"),
    ("reddit", "u/veteran_user_99", "Love how the 'new and improved' app is somehow slower than the 2019 version. Impressive engineering, honestly."),
    ("youtube", "CommentSectionCarl", "Ah yes, 'revolutionary' — it's a to-do list with a gradient background."),
    # --- Acute frustration ---
    ("twitter", "@priya_runs", "Third time this week the export button spins forever. This is unusable and I'm losing client work over it. FIX IT."),
    ("reddit", "u/locked_out_larry", "Locked out of my account for 6 days and support has ghosted every ticket. Absolutely furious right now."),
    ("bluesky", "nightowl.dev", "Sync broke again after the latest update. That's the second data-loss bug this month."),
    ("youtube", "FrustratedViewer", "Ads every 90 seconds now? The app is becoming unwatchable."),
    # --- Constructive feedback ---
    ("reddit", "u/pm_thoughts", "Solid product overall, but the mobile editor needs autosave. Lost a long draft twice now. Otherwise, keep it up."),
    ("youtube", "LearningLoop", "Great video! One suggestion: chapter timestamps would make this much easier to follow."),
    ("twitter", "@ux_hana", "Feature request: dark mode for the dashboard. My eyes at 2am would appreciate it."),
    # --- Pricing skepticism ---
    ("twitter", "@bootstrapped_ben", "$49/mo for this when the free-tier competitor does 90% of it? The math ain't mathing."),
    ("reddit", "u/frugal_founder", "The new pricing tiers feel like a cash grab. Paying more for features that used to be standard is a hard pass."),
    ("bluesky", "indiehacker.bsky.social", "Free tier limits dropped from 1k to 100 requests? That's a bold strategy."),
    # --- Neutral inquiries ---
    ("bluesky", "newhere.bsky.social", "Just joined after the migration wave. What should I know as a newcomer?"),
    ("twitter", "@api_ada", "Does the API support webhooks yet, or is it still polling only? Checking before I build an integration."),
    ("youtube", "CuriousCat", "What camera and mic setup is this? The quality is crisp."),
    ("reddit", "u/selfhost_sam", "How does this compare to the self-hosted alternatives? Curious about resource usage."),
    ("twitter", "@eu_compliance", "Does this work with EU data-residency requirements?"),
]


async def generate_mock_stream(
    delay_range: tuple[float, float] = (1.0, 3.0),
) -> AsyncIterator[RawComment]:
    """Yield RawComments forever, pausing a random delay between each."""
    while True:
        platform, author, text = random.choice(COMMENT_POOL)
        yield RawComment(
            id=f"mock-{uuid.uuid4().hex[:12]}",
            platform=platform,
            text=text,
            author_handle=author,
            timestamp=datetime.now(UTC),
        )
        await asyncio.sleep(random.uniform(*delay_range))
