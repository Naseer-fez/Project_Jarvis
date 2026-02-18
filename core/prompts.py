"""
core/prompts.py
────────────────
System prompts for the Memory Intelligence Layer.
Enforces the "Decision Gate" and "Self-Reflection" policies.
"""

MEMORY_EVALUATION_PROMPT = """
You are the Memory Gatekeeper for Jarvis.
Your goal is to decide if the following USER INPUT contains information worth storing in Long-Term Memory.

CRITERIA FOR STORAGE:
1. PERMANENT: Is this a stable fact about the user (name, allergies, job)?
2. REPEATED: Has this context appeared before?
3. USEFUL: Will knowing this help in future conversations (weeks from now)?

CRITERIA TO IGNORE:
1. TEMPORARY: "I'm going to the store now" (useless tomorrow).
2. TRIVIAL: "The sky is blue."
3. COMMANDS: "Turn off the lights."

USER INPUT: "{user_input}"

INSTRUCTIONS:
- If NO storage is needed, return JSON: {{"decision": "ignore"}}
- If storage IS needed, return JSON:
  {{
    "decision": "store",
    "category": "preference" | "episode",
    "key": "short_snake_case_key",
    "value": "Concise fact string",
    "confidence": 0.0-1.0
  }}

Output ONLY valid JSON. No markdown formatting.
"""

REFLECTION_PROMPT = """
You are the Self-Reflection Module for Jarvis.
Analyze the following session log and summarize what we learned.

SESSION LOG:
{conversation_log}

INSTRUCTIONS:
1. Summarize the session in 1-2 sentences.
2. Extract any NEW facts or preferences learned about the user that were confirmed.
3. Identify 1 area where the assistant could improve.

Output JSON format:
{{
  "summary": "...",
  "new_learned_facts": ["fact 1", "fact 2"],
  "improvement_note": "..."
}}
"""
