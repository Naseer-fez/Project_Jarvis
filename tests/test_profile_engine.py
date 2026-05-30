import json
from pathlib import Path
from core.profile import UserProfileEngine


def test_profile_engine_defaults(tmp_path):
    """Verify that UserProfileEngine initializes with expected defaults."""
    UserProfileEngine.PROFILE_PATH = tmp_path / "user_profile.json"
    engine = UserProfileEngine()
    
    assert engine.interaction_count == 0
    assert engine.get_system_prompt_injection() == "User: User. Style: casual. Level: intermediate."
    assert engine.get_communication_style() == "Be friendly and conversational. Keep it natural."


def test_profile_engine_save_load(tmp_path):
    """Verify that user profile can be saved to disk and reloaded correctly."""
    UserProfileEngine.PROFILE_PATH = tmp_path / "user_profile.json"
    
    engine = UserProfileEngine()
    engine._data["name"] = "Alice"
    engine._data["communication_style"] = "formal"
    engine._data["preferred_topics"] = ["AI", "Gaming"]
    engine.save()

    assert UserProfileEngine.PROFILE_PATH.exists()

    # Load into a new engine instance
    engine2 = UserProfileEngine()
    assert engine2._data["name"] == "Alice"
    assert engine2._data["communication_style"] == "formal"
    assert engine2._data["preferred_topics"] == ["AI", "Gaming"]
    assert engine2.get_system_prompt_injection() == "User: Alice. Style: formal. Level: intermediate. Interests: AI, Gaming."
    assert engine2.get_communication_style() == "Be precise and professional. Use formal language."


def test_profile_engine_update_from_conversation(tmp_path):
    """Verify that profile engine updates interaction statistics and extracts user names from conversation."""
    UserProfileEngine.PROFILE_PATH = tmp_path / "user_profile.json"
    engine = UserProfileEngine()

    engine.update_from_conversation("My name is John.", "Hello John!")
    assert engine.interaction_count == 1
    assert engine._data["name"] == "John"
    assert engine._data["first_seen"] is not None
    assert engine._data["last_seen"] is not None

    # Secondary update should increment count but not change name if no pattern matches
    engine.update_from_conversation("How is the weather?", "It is sunny.")
    assert engine.interaction_count == 2
    assert engine._data["name"] == "John"


def test_profile_engine_apply_delta(tmp_path):
    """Verify that synthesis delta updates are properly validated and applied."""
    UserProfileEngine.PROFILE_PATH = tmp_path / "user_profile.json"
    engine = UserProfileEngine()

    delta = {
        "name": {
            "value": "Bob",
            "confidence": 0.8
        },
        "communication_style": {
            "value": "technical",
            "confidence": 0.9
        },
        "expertise_level": {
            "value": "expert",
            "confidence": 0.3  # Too low confidence
        },
        "preferred_topics": {
            "value": ["Python", "C++"],
            "confidence": 0.9
        }
    }

    updated = engine.apply_delta(delta)
    assert "name" in updated
    assert "communication_style" in updated
    assert "expertise_level" not in updated
    assert "preferred_topics" in updated

    assert engine._data["name"] == "Bob"
    assert engine._data["communication_style"] == "technical"
    assert engine._data["expertise_level"] == "intermediate"  # Remained default
    assert engine._data["preferred_topics"] == ["Python", "C++"]
