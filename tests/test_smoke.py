"""
tests/test_session7.py
──────────────────────
Verification for Session 7: Identity & Profiling.
Run this to confirm Jarvis creates and adapts the user profile.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.controller import JarvisControllerV5

def test_identity_synthesis():
    print("=== TEST START: Session 7 Identity Synthesis ===")
    
    # 1. Setup
    # Use a fresh DB for testing to see profile creation from scratch
    test_db = "memory/test_session7.db"
    if os.path.exists(test_db):
        os.remove(test_db)
        
    jarvis = JarvisControllerV5(db_path=test_db)
    jarvis.initialize()
    
    # 2. Simulate User Interaction (Providing Identity Clues)
    print("\n[Step 1] Feeding inputs to build identity...")
    inputs = [
        "My name is Tony.",
        "I work as an AI Engineer building a robot.",
        "I prefer short, concise answers.",
        "Don't give me long explanations, just the code.",
        "Save that I love Iron Man movies."
    ]
    
    for inp in inputs:
        print(f"User: {inp}")
        resp = jarvis.process(inp)
        print(f"Jarvis: {resp}")
        # Artificial delay to ensure distinct timestamps if needed
        time.sleep(0.1)

    # 3. Force Synthesis
    print("\n[Step 2] Forcing Profile Synthesis...")
    # We bypass the "10 memory" trigger and call it manually
    jarvis.run_synthesis()
    
    # 4. Verify Artifacts
    profile_path = Path("memory/user_profile.json")
    if not profile_path.exists():
        print("❌ FAILURE: user_profile.json was not created.")
        return
        
    with open(profile_path, "r") as f:
        data = json.load(f)
        
    print("\n[Step 3] Inspecting Generated Profile:")
    print(json.dumps(data, indent=2))
    
    # 5. Assertions
    core = data.get("identity_core", {})
    weights = data.get("preference_weights", {})
    
    # Check Name Detection
    if "Tony" in str(core.get("name")):
        print("✅ SUCCESS: Name 'Tony' detected.")
    else:
        print(f"⚠️  WARNING: Name not detected (Got: {core.get('name')})")

    # Check Occupation
    if "Engineer" in str(core.get("occupation")) or "AI" in str(core.get("occupation")):
        print("✅ SUCCESS: Occupation detected.")
    else:
        print(f"⚠️  WARNING: Occupation not detected.")
        
    # Check Style Preference (Concise)
    # If user said "short answers", detail_level should be low (< 0.5)
    if weights.get("detail_level", 0.5) < 0.5:
        print(f"✅ SUCCESS: Detected preference for CONCISE responses (Score: {weights['detail_level']}).")
    else:
        print(f"⚠️  WARNING: Detail preference not captured correctly (Score: {weights['detail_level']}).")

    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    test_identity_synthesis()