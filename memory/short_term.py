"""
Short-term memory for session-scoped context.
Stores conversation context and temporary data in RAM.
"""

from typing import List, Dict, Any
from datetime import datetime
from collections import deque


class ShortTermMemory:
    """Manages in-memory session context."""
    
    def __init__(self, max_history: int = 20):
        """
        Initialize short-term memory.
        
        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.conversation_buffer = deque(maxlen=max_history)
        self.session_start = datetime.now()
        self.session_data = {}  # Temporary key-value storage
    
    def add_exchange(self, user_input: str, assistant_response: str):
        """
        Add a conversation exchange to the buffer.
        
        Args:
            user_input: User's message
            assistant_response: Assistant's response
        """
        self.conversation_buffer.append({
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': datetime.now()
        })
    
    def get_recent_context(self, num_turns: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation turns.
        
        Args:
            num_turns: Number of recent turns to retrieve
        
        Returns:
            List of recent conversation exchanges
        """
        return list(self.conversation_buffer)[-num_turns:]
    
    def get_formatted_context(self, num_turns: int = 5) -> str:
        """
        Get formatted conversation history for LLM context.
        
        Args:
            num_turns: Number of recent turns to include
        
        Returns:
            Formatted string of recent conversation
        """
        recent = self.get_recent_context(num_turns)
        
        formatted = []
        for exchange in recent:
            formatted.append(f"User: {exchange['user']}")
            formatted.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(formatted)
    
    def set_session_data(self, key: str, value: Any):
        """Store temporary session data."""
        self.session_data[key] = value
    
    def get_session_data(self, key: str, default: Any = None) -> Any:
        """Retrieve temporary session data."""
        return self.session_data.get(key, default)
    
    def clear(self):
        """Clear all short-term memory."""
        self.conversation_buffer.clear()
        self.session_data.clear()
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        return (datetime.now() - self.session_start).total_seconds()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session metadata
        """
        return {
            'start_time': self.session_start,
            'duration_seconds': self.get_session_duration(),
            'total_exchanges': len(self.conversation_buffer),
            'session_data_keys': list(self.session_data.keys())
        }


if __name__ == "__main__":
    # Test short-term memory
    print("Testing Short-Term Memory System...")
    
    stm = ShortTermMemory(max_history=5)
    
    print("\n[1] Adding conversation exchanges...")
    stm.add_exchange("Hello", "Hi! How can I help you?")
    stm.add_exchange("What's the weather?", "I don't have access to weather data yet.")
    stm.add_exchange("Remember I like coffee", "I've stored that you like coffee.")
    
    print("[2] Getting recent context...")
    context = stm.get_recent_context(num_turns=2)
    for exchange in context:
        print(f"  User: {exchange['user']}")
        print(f"  Assistant: {exchange['assistant']}")
        print()
    
    print("[3] Getting formatted context for LLM...")
    formatted = stm.get_formatted_context(num_turns=3)
    print(formatted)
    
    print("\n[4] Session summary...")
    summary = stm.get_session_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Short-term memory test complete")

