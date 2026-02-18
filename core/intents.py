"""
Intent classification system.
Routes user input to appropriate handlers.
"""

import re
from typing import Tuple, Optional, Dict
from enum import Enum


class Intent(Enum):
    """Possible user intent types."""
    MEMORY_STORE = "memory_store"      # User wants to store a preference/fact
    MEMORY_RECALL = "memory_recall"    # User wants to recall stored information
    QUESTION = "question"              # General question
    COMMAND = "command"                # System command (exit, help, etc.)
    UNKNOWN = "unknown"                # Cannot determine intent


class IntentClassifier:
    """Classifies user input into intent categories."""
    
    # Pattern definitions
    MEMORY_STORE_PATTERNS = [
        r"^remember (that )?(.+)",
        r"^i (like|prefer|enjoy|hate|dislike) (.+)",
        r"^my (.+) is (.+)",
        r"^store (that )?(.+)",
        r"^save (that )?(.+)",
    ]
    
    MEMORY_RECALL_PATTERNS = [
        r"^what (do|did) (i|you know about) (.+)",
        r"^do i (like|prefer|enjoy) (.+)",
        r"^what('s| is) my (.+)",
        r"^tell me about my (.+)",
        r"^recall (.+)",
    ]
    
    COMMAND_PATTERNS = {
        'exit': [r"^(exit|quit|bye|goodbye)$"],
        'help': [r"^help$", r"^what can you do"],
        'clear': [r"^clear( memory)?$"],
        'status': [r"^status$", r"^how are you"],
    }
    
    def __init__(self):
        pass
    
    def classify(self, user_input: str) -> Intent:
        if not user_input or not user_input.strip():
            return Intent.UNKNOWN
        
        normalized = user_input.lower().strip()
        
        # Check for commands first
        for command_type, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, normalized, re.IGNORECASE):
                    return Intent.COMMAND
        
        # Check for memory storage
        for pattern in self.MEMORY_STORE_PATTERNS:
            if re.match(pattern, normalized, re.IGNORECASE):
                return Intent.MEMORY_STORE
        
        # Check for memory recall
        for pattern in self.MEMORY_RECALL_PATTERNS:
            if re.match(pattern, normalized, re.IGNORECASE):
                return Intent.MEMORY_RECALL
        
        return Intent.QUESTION
    
    def extract_memory_data(self, user_input: str) -> Optional[Dict[str, str]]:
        """Extract key-value pair from memory storage intent."""
        normalized = user_input.lower().strip()
        
        # Pattern: "remember [that] X"
        match = re.match(r"remember (?:that )?(.+)", normalized, re.IGNORECASE)
        if match:
            content = match.group(1)
            if content.startswith("i "):
                content = content[2:]
                if content.startswith("like "):
                    return {"favorite_thing": content[5:]}
                elif content.startswith("prefer "):
                    return {"preference": content[7:]}
                elif content.startswith("enjoy "):
                    return {"enjoyment": content[6:]}
                elif content.startswith("hate ") or content.startswith("dislike "):
                    verb_len = 5 if content.startswith("hate") else 8
                    return {"dislike": content[verb_len:]}
            return {"fact": content}
        
        # Pattern: "I like/prefer X"
        match = re.match(r"i (like|prefer|enjoy) (.+)", normalized, re.IGNORECASE)
        if match:
            verb = match.group(1)
            value = match.group(2)
            key = "favorite_thing" if verb == "like" else "preference"
            return {key: value}
        
        # Pattern: "my X is Y"
        match = re.match(r"my (.+?) is (.+)", normalized, re.IGNORECASE)
        if match:
            key = match.group(1).replace(" ", "_")
            value = match.group(2)
            return {key: value}
        
        return None
    
    def extract_memory_query(self, user_input: str) -> Optional[str]:
        """Extract what the user wants to recall."""
        normalized = user_input.lower().strip()
        
        match = re.match(r"what do i (like|prefer|enjoy) (?:to )?(.+)", normalized)
        if match: return match.group(2).rstrip("?")
        
        match = re.match(r"do i (like|prefer|enjoy) (.+)", normalized)
        if match: return match.group(2).rstrip("?")
        
        match = re.match(r"what(?:'s| is) my (.+)", normalized)
        if match: return match.group(1).rstrip("?")
        
        match = re.match(r"tell me about my (.+)", normalized)
        if match: return match.group(1).rstrip("?")
        
        # Fallback for explicit recall command
        if normalized.startswith("recall "):
            return normalized[7:]
            
        return None

# --- API CONTRACT ENFORCEMENT (FIX FOR CONTROLLER) ---
# These functions wrap the class logic so controller.py can import them directly.

_classifier = IntentClassifier()

def classify_intent(user_input: str) -> Intent:
    return _classifier.classify(user_input)

def extract_memory_data(user_input: str) -> Optional[Dict[str, str]]:
    return _classifier.extract_memory_data(user_input)

def extract_memory_query(user_input: str) -> Optional[str]:
    return _classifier.extract_memory_query(user_input)