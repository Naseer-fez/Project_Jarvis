"""
core/tools/vision.py
────────────────────
Multimodal vision capabilities using LLaVA via Ollama.
Allows Jarvis to "see" images provided via file path.
"""

import base64
import logging
from pathlib import Path
import httpx

logger = logging.getLogger("Jarvis.Vision")

VISION_MODEL = "llava"  # Ensure you run: ollama pull llava

async def analyze_image(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    """
    Analyzes an image file using the LLaVA vision model.
    
    Args:
        image_path: Local path to the image file (jpg/png).
        prompt: What to look for or ask about the image.
    """
    path = Path(image_path).resolve()
    
    # 1. Validation
    if not path.exists():
        return f"Error: Image file not found at {image_path}"
    
    if path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        return "Error: Only .jpg and .png files are supported."

    # 2. Encode Image
    try:
        with open(path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        return f"Error reading image: {e}"

    # 3. Query Ollama (LLaVA)
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }

    logger.info(f"Sending image {path.name} to {VISION_MODEL}...")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("http://localhost:11434/api/generate", json=payload)
            resp.raise_for_status()
            result = resp.json()
            description = result.get("response", "").strip()
            return f"[Vision Analysis] {description}"
            
    except Exception as e:
        logger.error(f"Vision API failed: {e}")
        return f"Failed to analyze image. Ensure 'ollama serve' is running and you have pulled the model (ollama pull {VISION_MODEL}). Error: {e}"