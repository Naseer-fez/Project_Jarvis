"""
core/controller_v2.py
───────────────────
Orchestration layer for Jarvis Session 4.
"""
import uuid
from core.memory.hybrid_memory import HybridMemory
from core.llm.llm_v2 import LLMClientV2

class JarvisControllerV2:
    def __init__(self, config=None, voice=False, db_path="memory/memory.db", chroma_path="data/chroma", model_name="deepseek-r1:8b", embedding_model="all-MiniLM-L6-v2"):
        self.voice_enabled = voice
        if config is not None:
            import configparser
            if isinstance(config, configparser.ConfigParser):
                db_path = config.get("memory", "db_path", fallback=db_path)
                chroma_path = config.get("memory", "chroma_path", fallback=chroma_path)
                model_name = config.get("llm", "model", fallback=model_name)
        self.session_id = __import__("uuid").uuid4().hex[:8]
        self.memory = HybridMemory(db_path=db_path, chroma_path=chroma_path, model_name=embedding_model)
        self.llm = LLMClientV2(model_name=model_name)
        self.exchanges = 0

    def initialize(self):
        mem_status = self.memory.initialize()
        return {"session_id": self.session_id, "memory_mode": mem_status.get("mode")}

    def process(self, user_input: str) -> str:
        self.exchanges += 1
        ui_lower = user_input.lower().strip()

        # Hardcoded commands mapped for testing
        if ui_lower == "status":
            return f"Session: {self.session_id} | Memory Mode: {self.memory.mode}"
        if ui_lower == "help":
            return "Commands: status, exit, remember <fact>, what's <query>"

        # Intent Detection: Store
        if ui_lower.startswith("remember i like "):
            item = user_input[16:].strip()
            self.memory.store_preference(f"likes_{item[:5]}", item)
            return f"✓ I'll remember you like {item}"
            
        if ui_lower.startswith("my name is "):
            name = user_input[11:].strip()
            self.memory.store_preference("name", name)
            return f"✓ I'll remember your name is {name}"

        if ui_lower.startswith("i prefer "):
            item = user_input[9:].strip()
            self.memory.store_preference(f"prefer_{item[:5]}", item)
            return f"✓ I'll remember you prefer {item}"

        if ui_lower.startswith("i work in "):
            item = user_input[10:].strip()
            self.memory.store_preference("work", item)
            return f"✓ I'll remember you work in {item}"

        # Intent Detection: Recall / Question
        context = self.memory.build_context_block(user_input)
        
        # Check LLM offline fallback (test suite safety)
        response = self.llm.chat(user_input, system_context=context)
        if response == "LLM Offline.":
            prefs = self.memory.recall_preferences(user_input, top_k=1)
            if prefs and prefs[0]['value']:
                return f"Offline Fallback found in memory: {prefs[0]['value']}"
            return "I don't know (Offline)."

        self.memory.store_conversation(user_input, response, self.session_id)
        return response

    def session_summary(self):
        return {"session_id": self.session_id, "exchanges": self.exchanges}




    async def start(self):
        """Initialize and start the controller."""
        self.initialize()

    async def run_cli(self):
        """Interactive CLI loop."""
        import asyncio
        print(f"Jarvis V2 ready (session {self.session_id}). Type 'exit' to quit.")
        loop = asyncio.get_event_loop()
        while True:
            try:
                user_input = await loop.run_in_executor(None, input, "You: ")
            except EOFError:
                break
            if user_input.strip().lower() in ("exit", "quit"):
                break
            response = self.process(user_input)
            print(f"Jarvis: {response}")

    async def shutdown(self):
        """Graceful shutdown."""
        pass

# Alias for backward-compatible import in main.py
Controller = JarvisControllerV2