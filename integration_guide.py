"""
=======================================================================
  SESSION 5 – VOICE LAYER INTEGRATION
  Paste the snippets below into the indicated files.
=======================================================================

────────────────────────────────────────────────────────────────────────
FILE 1:  jarvis/main_v2.py   (or scripts/main_v2.py – whichever is your
         actual entry-point)
────────────────────────────────────────────────────────────────────────

Replace / augment your existing async main() with the version below.
The only changes are:
  • import VoiceLayer
  • call voice_layer.start() after the controller is ready
  • pass voice_layer into the controller so it can call ask_confirm()
  • call voice_layer.stop() on shutdown
"""

# ── main_v2.py  (diff-style – ADD these lines) ──────────────────────

MAIN_V2_ADDITIONS = '''
import asyncio
import configparser
from pathlib import Path

# --- NEW IMPORT ---
from core.voice.voice_layer import VoiceLayer

# Assuming your controller lives here (adjust if different):
from core.controller_v2 import JarvisController


CONFIG_PATH = Path("config/jarvis.ini")


async def main() -> None:
    # ── Load config ───────────────────────────────────────────────────
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)

    # ── Boot controller ───────────────────────────────────────────────
    controller = JarvisController(cfg)
    await controller.start()

    # ── Boot voice layer ──────────────────────────────────────────────
    # text_handler wraps the controller so the voice layer can call it.
    async def text_handler(user_text: str) -> str:
        """Bridge: voice text → controller → response text."""
        return await controller.process_input(user_text)

    voice_layer = VoiceLayer(cfg, text_handler)
    await voice_layer.start()

    # Inject voice_layer into controller so agentic hooks can call
    # ask_confirm() without importing voice_layer directly.
    controller.voice_layer = voice_layer          # see controller patch below

    # ── Run until interrupted ─────────────────────────────────────────
    try:
        # Keep the event loop alive; the wake-word thread drives interactions.
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await voice_layer.stop()
        await controller.stop()


if __name__ == "__main__":
    asyncio.run(main())
'''

# ────────────────────────────────────────────────────────────────────────
# FILE 2:  core/controller_v2.py
#
# Only ONE method needs a small addition: wherever your controller
# handles the REQUIRE_CONFIRM signal from AutonomyPolicy.
#
# Search for the place where you check the policy result, and add
# the voice confirmation hook as shown below.
# ────────────────────────────────────────────────────────────────────────

CONTROLLER_V2_PATCH = '''
# At the top of controller_v2.py, add:
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from core.voice.voice_layer import VoiceLayer


class JarvisController:
    def __init__(self, cfg):
        # ... your existing __init__ code ...
        self.voice_layer: Optional["VoiceLayer"] = None   # injected by main_v2


    # ── Inside whatever method runs a mission step ────────────────────
    # Find the block that looks like:
    #
    #     decision = self.autonomy_policy.evaluate(step)
    #     if decision == "REQUIRE_CONFIRM":
    #         # currently: probably prints to console or raises
    #         ...
    #
    # Replace / augment it with:

    async def _check_confirm(self, question: str) -> bool:
        """
        Ask for confirmation via voice (if available) or console fallback.
        Returns True if the user approved, False otherwise.
        """
        if self.voice_layer is not None:
            result = await self.voice_layer.ask_confirm(question)
            # result: True=yes, False=no, None=abort
            return result is True
        else:
            # Console fallback (original behaviour)
            ans = input(f"[CONFIRM REQUIRED] {question} (yes/no): ").strip().lower()
            return ans.startswith("y")

    # Example usage inside your mission-step executor:
    #
    #   if policy_decision == "REQUIRE_CONFIRM":
    #       approved = await self._check_confirm(
    #           f"Mission step '{step.name}' is high-risk. Proceed?"
    #       )
    #       if not approved:
    #           step.abort()
    #           return
'''

# ────────────────────────────────────────────────────────────────────────
# FILE 3:  core/agentic/scheduler.py  (NO internal changes)
#
# The AgentScheduler.tick() keeps running independently on its own
# asyncio.Task.  The voice layer sits in parallel; both share the same
# event loop.  No changes needed here.
# ────────────────────────────────────────────────────────────────────────

# Print instructions when run directly
if __name__ == "__main__":
    print(__doc__)
    print(MAIN_V2_ADDITIONS)
    print(CONTROLLER_V2_PATCH)
