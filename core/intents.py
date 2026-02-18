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
        """Initialize intent classifier."""
        pass
    
    def classify(self, user_input: str) -> Intent:
        """
        Classify user input into an intent.
        
        Args:
            user_input: User's message (normalized to lowercase)
        
        Returns:
            Detected intent
        """
        if not user_input or not user_input.strip():
            return Intent.UNKNOWN
        
        normalized = user_input.lower().strip()
        
        # Check for commands first (highest priority)
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
        
        # Default to question
        return Intent.QUESTION
    
    def extract_memory_data(self, user_input: str) -> Optional[Tuple[str, str]]:
        """
        Extract key-value pair from memory storage intent.
        
        Args:
            user_input: User's message
        
        Returns:
            Tuple of (key, value) or None if cannot extract
        
        Examples:
            "remember I like coffee" -> ("favorite_drink", "coffee")
            "my name is John" -> ("name", "John")
            "I prefer direct communication" -> ("communication_style", "direct communication")
        """
        normalized = user_input.lower().strip()
        
        # Pattern: "remember [that] X"
        match = re.match(r"remember (?:that )?(.+)", normalized, re.IGNORECASE)
        if match:
            content = match.group(1)
            # Try to split into key-value
            # "I like coffee" -> favorite_drink: coffee
            if content.startswith("i "):
                content = content[2:]  # Remove "i "
                if content.startswith("like "):
                    return ("favorite_thing", content[5:])
                elif content.startswith("prefer "):
                    return ("preference", content[7:])
                elif content.startswith("enjoy "):
                    return ("enjoyment", content[6:])
                elif content.startswith("hate ") or content.startswith("dislike "):
                    verb_len = 5 if content.startswith("hate") else 8
                    return ("dislike", content[verb_len:])
            return ("fact", content)
        
        # Pattern: "I like/prefer X"
        match = re.match(r"i (like|prefer|enjoy) (.+)", normalized, re.IGNORECASE)
        if match:
            verb = match.group(1)
            value = match.group(2)
            if verb == "like":
                return ("favorite_thing", value)
            elif verb == "prefer":
                return ("preference", value)
            else:
                return ("enjoyment", value)
        
        # Pattern: "I hate/dislike X"
        match = re.match(r"i (hate|dislike) (.+)", normalized, re.IGNORECASE)
        if match:
            value = match.group(2)
            return ("dislike", value)
        
        # Pattern: "my X is Y"
        match = re.match(r"my (.+?) is (.+)", normalized, re.IGNORECASE)
        if match:
            key = match.group(1).replace(" ", "_")
            value = match.group(2)
            return (key, value)
        
        return None
    
    def extract_memory_query(self, user_input: str) -> Optional[str]:
        """
        Extract what the user wants to recall.
        
        Args:
            user_input: User's message
        
        Returns:
            Search term or None
        
        Examples:
            "what do I like to drink?" -> "drink"
            "what's my name?" -> "name"
        """
        normalized = user_input.lower().strip()
        
        # Pattern: "what do I like/prefer X"
        match = re.match(r"what do i (like|prefer|enjoy) (?:to )?(.+)", normalized)
        if match:
            return match.group(2).rstrip("?")
        
        # Pattern: "do I like X"
        match = re.match(r"do i (like|prefer|enjoy) (.+)", normalized)
        if match:
            return match.group(2).rstrip("?")
        
        # Pattern: "what's my X"
        match = re.match(r"what(?:'s| is) my (.+)", normalized)
        if match:
            return match.group(1).rstrip("?")
        
        # Pattern: "tell me about my X"
        match = re.match(r"tell me about my (.+)", normalized)
        if match:
            return match.group(1).rstrip("?")
        
        return None
    
    def get_command_type(self, user_input: str) -> Optional[str]:
        """
        Get specific command type if input is a command.
        
        Args:
            user_input: User's message
        
        Returns:
            Command name ('exit', 'help', etc.) or None
        """
        normalized = user_input.lower().strip()
        
        for command_type, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, normalized, re.IGNORECASE):
                    return command_type
        
        return None


if __name__ == "__main__":
    # Test intent classifier
    print("Testing Intent Classifier...")
    
    classifier = IntentClassifier()
    
    test_cases = [
        ("remember I like coffee", Intent.MEMORY_STORE),
        ("my name is John", Intent.MEMORY_STORE),
        ("I prefer direct communication", Intent.MEMORY_STORE),
        ("what do I like to drink?", Intent.MEMORY_RECALL),
        ("what's my name?", Intent.MEMORY_RECALL),
        ("do I like coffee?", Intent.MEMORY_RECALL),
        ("what is the weather?", Intent.QUESTION),
        ("exit", Intent.COMMAND),
        ("help", Intent.COMMAND),
    ]
    
    print("\n[1] Testing intent classification...")
    for text, expected in test_cases:
        detected = classifier.classify(text)
        status = "✓" if detected == expected else "✗"
        print(f"  {status} '{text}' -> {detected.value} (expected: {expected.value})")
    
    print("\n[2] Testing memory data extraction...")
    memory_tests = [
        "remember I like coffee",
        "my name is John",
        "I prefer direct communication",
        "I hate waking up early",
    ]
    
    for text in memory_tests:
        data = classifier.extract_memory_data(text)
        print(f"  '{text}'")
        print(f"    -> {data}")
    
    print("\n[3] Testing memory query extraction...")
    query_tests = [
        "what do I like to drink?",
        "what's my name?",
        "do I prefer coffee?",
    ]
    
    for text in query_tests:
        query = classifier.extract_memory_query(text)
        print(f"  '{text}' -> '{query}'")
    
    print("\n✓ Intent classifier test complete")
