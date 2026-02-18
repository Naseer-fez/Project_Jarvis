"""
Long-term memory persistence using SQLite.
Stores user preferences, facts, and episodic events.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class LongTermMemory:
    """Manages persistent storage of user data and episodic events."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize long-term memory database.
        
        Args:
            db_path: Path to SQLite database file. 
                     Defaults to memory/memory.db
        """
        if db_path is None:
            # Use relative path from project root
            db_path = Path(__file__).parent / "memory.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._init_tables()
    
    def _init_tables(self):
        """Create tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Episodic memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category TEXT
            )
        """)
        
        # Conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        """)
        
        self.conn.commit()
    
    # === PREFERENCES ===
    
    def store_preference(self, key: str, value: str) -> bool:
        """
        Store or update a user preference.
        
        Args:
            key: Preference identifier (e.g., 'favorite_drink')
            value: Preference value (e.g., 'coffee')
        
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO preferences (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
            """, (key, value, datetime.now()))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error storing preference: {e}")
            return False
    
    def get_preference(self, key: str) -> Optional[str]:
        """
        Retrieve a specific preference.
        
        Args:
            key: Preference identifier
        
        Returns:
            Preference value or None if not found
        """
        cursor = self.conn.cursor()
        result = cursor.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        
        return result['value'] if result else None
    
    def get_all_preferences(self) -> Dict[str, str]:
        """
        Get all stored preferences.
        
        Returns:
            Dictionary of key-value pairs
        """
        cursor = self.conn.cursor()
        results = cursor.execute(
            "SELECT key, value FROM preferences ORDER BY updated_at DESC"
        ).fetchall()
        
        return {row['key']: row['value'] for row in results}
    
    def delete_preference(self, key: str) -> bool:
        """Delete a preference."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM preferences WHERE key = ?", (key,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting preference: {e}")
            return False
    
    # === EPISODIC MEMORY ===
    
    def store_event(self, event: str, category: str = None) -> bool:
        """
        Store an episodic memory event.
        
        Args:
            event: Description of the event
            category: Optional category (e.g., 'project', 'conversation')
        
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO episodic_memory (event, category, timestamp)
                VALUES (?, ?, ?)
            """, (event, category, datetime.now()))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error storing event: {e}")
            return False
    
    def get_recent_events(self, limit: int = 10, category: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve recent episodic events.
        
        Args:
            limit: Maximum number of events to return
            category: Optional category filter
        
        Returns:
            List of event dictionaries
        """
        cursor = self.conn.cursor()
        
        if category:
            results = cursor.execute("""
                SELECT event, category, timestamp
                FROM episodic_memory
                WHERE category = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (category, limit)).fetchall()
        else:
            results = cursor.execute("""
                SELECT event, category, timestamp
                FROM episodic_memory
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        return [dict(row) for row in results]
    
    # === CONVERSATION HISTORY ===
    
    def store_conversation(self, user_input: str, response: str, session_id: str = None) -> bool:
        """
        Store a conversation exchange.
        
        Args:
            user_input: User's message
            response: Assistant's response
            session_id: Optional session identifier
        
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO conversation_history 
                (user_input, assistant_response, session_id, timestamp)
                VALUES (?, ?, ?, ?)
            """, (user_input, response, session_id, datetime.now()))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error storing conversation: {e}")
            return False
    
    def get_conversation_history(self, limit: int = 20, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history.
        
        Args:
            limit: Maximum number of exchanges to return
            session_id: Optional session filter
        
        Returns:
            List of conversation dictionaries
        """
        cursor = self.conn.cursor()
        
        if session_id:
            results = cursor.execute("""
                SELECT user_input, assistant_response, timestamp
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit)).fetchall()
        else:
            results = cursor.execute("""
                SELECT user_input, assistant_response, timestamp
                FROM conversation_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        return [dict(row) for row in results]
    
    # === UTILITY ===
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test the memory system
    print("Testing Long-Term Memory System...")
    
    with LongTermMemory() as memory:
        # Test preferences
        print("\n[1] Storing preferences...")
        memory.store_preference("favorite_drink", "coffee")
        memory.store_preference("work_style", "night owl")
        memory.store_preference("communication_style", "direct and concise")
        
        print("[2] Retrieving preference...")
        drink = memory.get_preference("favorite_drink")
        print(f"Favorite drink: {drink}")
        
        print("[3] Getting all preferences...")
        all_prefs = memory.get_all_preferences()
        for key, value in all_prefs.items():
            print(f"  {key}: {value}")
        
        # Test episodic memory
        print("\n[4] Storing episodic events...")
        memory.store_event("Built Jarvis memory system", category="project")
        memory.store_event("Implemented SQLite persistence", category="project")
        
        print("[5] Retrieving recent events...")
        events = memory.get_recent_events(limit=5)
        for event in events:
            print(f"  [{event['timestamp']}] {event['event']}")
        
        # Test conversation history
        print("\n[6] Storing conversation...")
        memory.store_conversation(
            user_input="Remember I like coffee",
            response="I've stored that you like coffee.",
            session_id="test-session-1"
        )
        
        print("[7] Retrieving conversation history...")
        history = memory.get_conversation_history(limit=5)
        for exchange in history:
            print(f"  User: {exchange['user_input']}")
            print(f"  Assistant: {exchange['assistant_response']}")
            print()
    
    print("✓ Memory system test complete")

