"""
core/tools/vision.py
═════════════════════
Vision ingestion using LLaVA via Ollama.

V1 Rules:
  - Passive. Read-only. Explicitly requested only.
  - NOT continuous. NOT autonomous. NOT tied to actions.
  - Returns: description + confidence
  - Stores output to logs (never to action queue)
  - Allowed prompts are whitelisted — no "what should I do?" queries
  - If image is ambiguous → say so and ask user. Never guess.

Hard NO in V1:
  ❌ "What should I do with this?"
  ❌ "What's the next physical step?"
  ❌ Continuous camera loop
  ❌ Triggering any action from vision output
"""

import asyncio
import aiohttp
import base64
from pathlib import Path
from datetime import datetime, timezone
from core.logger import get_logger, audit

logger = get_logger("vision")

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"

# ══════════════════════════════════════════════
# Whitelisted prompts only — no action-inducing queries
# ══════════════════════════════════════════════
ALLOWED_PROMPTS = {
    "describe": "Describe what is visible in this image. Be objective and factual.",
    "objects": "What objects are present in this image? List them clearly.",
    "clarity": "Is this image clear or ambiguous? Describe any unclear areas.",
    "text": "What text is visible in this image? Transcribe exactly what you see.",
    "scene": "Describe the scene or environment shown in this image.",
}

# Prompts that are BLOCKED — these could lead to action reasoning
BLOCKED_PROMPT_KEYWORDS = [
    "what should i do",
    "next step",
    "how do i act",
    "what action",
    "should i click",
    "move to",
    "go to",
]


def _is_prompt_safe(prompt: str) -> bool:
    lower = prompt.lower()
    return not any(kw in lower for kw in BLOCKED_PROMPT_KEYWORDS)


def _load_image_as_base64(image_path: str) -> str | None:
    path = Path(image_path)
    if not path.exists():
        logger.error(f"Image not found: {image_path}")
        return None
    if not path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        logger.error(f"Unsupported image format: {path.suffix}")
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VisionTool:
    """
    Passive vision analysis tool.
    Accepts image path or base64 string.
    Returns description and confidence estimate.
    """

    async def analyze_image(
        self,
        image_path: str,
        prompt_key: str = "describe",
        custom_prompt: str | None = None,
    ) -> dict:
        """
        Analyze an image using LLaVA.

        Args:
            image_path: Path to image file
            prompt_key: Key from ALLOWED_PROMPTS dict
            custom_prompt: Custom prompt (safety-checked)

        Returns:
            {
                "success": bool,
                "description": str,
                "confidence": str,  # "high" | "medium" | "low" | "ambiguous"
                "prompt_used": str,
                "image": str,
                "ts": str,
                "error": str | None
            }
        """
        ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"VISION: analyzing image={image_path!r} prompt={prompt_key!r}")

        # Resolve prompt
        if custom_prompt:
            if not _is_prompt_safe(custom_prompt):
                return self._blocked_result(image_path, ts, "Custom prompt contains action-inducing keywords. Blocked.")
            prompt_text = custom_prompt
        elif prompt_key in ALLOWED_PROMPTS:
            prompt_text = ALLOWED_PROMPTS[prompt_key]
        else:
            return self._blocked_result(
                image_path, ts,
                f"Prompt key '{prompt_key}' not in allowed list: {list(ALLOWED_PROMPTS.keys())}"
            )

        # Load image
        b64 = _load_image_as_base64(image_path)
        if b64 is None:
            return self._error_result(image_path, ts, f"Could not load image: {image_path}")

        # Call LLaVA
        description = await self._call_llava(b64, prompt_text)
        if description is None:
            return self._error_result(image_path, ts, "LLaVA call failed or timed out")

        # Estimate confidence
        confidence = self._estimate_confidence(description)

        result = {
            "success": True,
            "description": description,
            "confidence": confidence,
            "prompt_used": prompt_text,
            "image": image_path,
            "ts": ts,
            "error": None,
        }

        audit(
            logger,
            f"VISION_COMPLETE: image={image_path!r} confidence={confidence}",
            tool="vision.analyze_image",
            action="vision_complete"
        )

        # Log warning if ambiguous
        if confidence == "ambiguous":
            logger.warning(
                f"VISION AMBIGUOUS: Image '{image_path}' produced uncertain results. "
                f"Recommend asking user for clarification."
            )

        return result

    async def _call_llava(self, b64_image: str, prompt: str) -> str | None:
        """Call LLaVA via Ollama. Returns response text or None on failure."""
        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "options": {
                "temperature": 0.1,
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    OLLAMA_URL, json=payload,
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"LLaVA call failed: HTTP {resp.status}")
                        return None
                    data = await resp.json()
                    return data.get("response", "").strip()
        except aiohttp.ClientConnectorError:
            logger.error("Ollama not reachable for LLaVA. Is it running? (ollama serve)")
            return None
        except asyncio.TimeoutError:
            logger.error("LLaVA call timed out after 90s")
            return None
        except Exception as e:
            logger.error(f"LLaVA exception: {e}")
            return None

    def _estimate_confidence(self, description: str) -> str:
        """
        Simple heuristic confidence from response content.
        Not ML — just keyword detection.
        """
        lower = description.lower()
        ambiguous_signals = [
            "unclear", "difficult to", "hard to tell", "cannot determine",
            "not sure", "ambiguous", "blurry", "low quality", "i don't know",
            "uncertain", "might be", "possibly", "perhaps"
        ]
        if any(sig in lower for sig in ambiguous_signals):
            return "ambiguous"
        if len(description) > 200:
            return "high"
        if len(description) > 80:
            return "medium"
        return "low"

    def _blocked_result(self, image: str, ts: str, reason: str) -> dict:
        logger.warning(f"VISION BLOCKED: {reason}")
        return {
            "success": False, "description": "", "confidence": "blocked",
            "prompt_used": "", "image": image, "ts": ts, "error": reason
        }

    def _error_result(self, image: str, ts: str, reason: str) -> dict:
        logger.error(f"VISION ERROR: {reason}")
        return {
            "success": False, "description": "", "confidence": "error",
            "prompt_used": "", "image": image, "ts": ts, "error": reason
        }
