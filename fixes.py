import re

filepath = 'core/llm/cloud_client.py'
with open(filepath, 'r') as f:
    content = f.read()

# 1. Update PROVIDERS list to prioritize Gemini
content = re.sub(
    r'PROVIDERS = \[.*?\]',
    'PROVIDERS = ["gemini", "groq", "openai", "anthropic"]',
    content
)

# 2. Add GEMINI_API_KEY to the provider keys check
content = re.sub(
    r'"groq": "GROQ_API_KEY",',
    '"gemini": "GEMINI_API_KEY",\n            "groq": "GROQ_API_KEY",',
    content
)

# 3. Route the call to _call_gemini in the _call method
call_injection = """if provider == "gemini":
            return await self._call_gemini(prompt, system, temperature)
        if provider == "groq":"""
content = re.sub(
    r'if provider == "groq":',
    call_injection,
    content
)

# 4. Implement the _call_gemini async method
gemini_method = """
    async def _call_gemini(self, prompt: str, system: str, temperature: float) -> str:
        import aiohttp
        api_key = os.environ["GEMINI_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature}
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
                try:
                    return str(data["candidates"][0]["content"]["parts"][0]["text"])
                except (KeyError, IndexError):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug("Gemini response missing content: %s", data)
                    return ""
"""

# Append the new method just before the __all__ export
content = content.replace('__all__ = ["CloudLLMClient"]', gemini_method + '\n__all__ = ["CloudLLMClient"]')

with open(filepath, 'w') as f:
    f.write(content)

print("Successfully updated core/llm/cloud_client.py to include Gemini support.")


# Run the patch script and then clean it up
