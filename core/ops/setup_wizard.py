"""
Interactive CLI Configuration Wizard for Jarvis.
Guides users through setting up external API keys and cloud fallback integrations.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_dict() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    
    env_vars = {}
    content = ENV_FILE.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            # Remove optional quotes surrounding value
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            env_vars[key.strip()] = val
    return env_vars


def save_env_dict(env_vars: dict[str, str]) -> None:
    # Read existing file to preserve comments/order if possible
    lines = []
    if ENV_FILE.exists():
        content = ENV_FILE.read_text(encoding="utf-8")
        lines = content.splitlines()
    
    updated_keys = set()
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in env_vars:
                new_lines.append(f"{key}=\"{env_vars[key]}\"")
                updated_keys.add(key)
                continue
        new_lines.append(line)
    
    # Append any keys that weren't already in the file
    for key, val in env_vars.items():
        if key not in updated_keys:
            new_lines.append(f"{key}=\"{val}\"")
            
    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def prompt_key(name: str, description: str, current_val: str = "", validation_pattern: str | None = None) -> str:
    print(f"\n* {name} ({description})")
    if current_val:
        # Redact secrets in display
        display_val = current_val
        if len(current_val) > 8:
            display_val = current_val[:4] + "..." + current_val[-4:]
        prompt = f"  Current: {display_val}\n  Enter new value (or press Enter to keep current, space to clear): "
    else:
        prompt = "  Enter value (or press Enter to skip): "
        
    choice = input(prompt).strip()
    if not choice:
        return current_val
    if choice == " ":
        return ""
        
    if validation_pattern and not re.match(validation_pattern, choice):
        print(f"  [Warning] Value does not match expected format ({validation_pattern}).")
        confirm = input("  Save anyway? (y/n): ").strip().lower()
        if confirm != "y":
            return prompt_key(name, description, current_val, validation_pattern)
            
    return choice


def main() -> None:
    print("====================================================")
    print("        JARVIS INTEGRATIONS SETUP WIZARD            ")
    print("====================================================")
    print("This wizard will help you configure external APIs.")
    print("Press Enter to skip/keep defaults. Type space to clear a value.")
    
    env_vars = load_env_dict()
    
    # 1. Google Gemini (Primary Fallback LLM)
    env_vars["GEMINI_API_KEY"] = prompt_key(
        "GEMINI_API_KEY",
        "Google Gemini AI Studio API key (free tier available at https://aistudio.google.com/)",
        env_vars.get("GEMINI_API_KEY", ""),
        r"^AIzaSy[A-Za-z0-9_-]{32,45}$"
    )
    
    # 2. Groq (Fast alternative LLM)
    env_vars["GROQ_API_KEY"] = prompt_key(
        "GROQ_API_KEY",
        "Groq API Key (https://console.groq.com/keys)",
        env_vars.get("GROQ_API_KEY", ""),
        r"^gsk_[A-Za-z0-9]{40,60}$"
    )
    
    # 3. Telegram Bot integration
    env_vars["TELEGRAM_BOT_TOKEN"] = prompt_key(
        "TELEGRAM_BOT_TOKEN",
        "Telegram Bot API Token (from @BotFather on Telegram)",
        env_vars.get("TELEGRAM_BOT_TOKEN", ""),
        r"^\d+:[A-Za-z0-9_-]{30,45}$"
    )
    if env_vars.get("TELEGRAM_BOT_TOKEN"):
        env_vars["TELEGRAM_CHAT_ID"] = prompt_key(
            "TELEGRAM_CHAT_ID",
            "Your numeric Telegram Chat ID (obtain via BotFather/getUpdates)",
            env_vars.get("TELEGRAM_CHAT_ID", ""),
            r"^-?\d+$"
        )
        
    # 4. Spotify Integration
    env_vars["SPOTIFY_CLIENT_ID"] = prompt_key(
        "SPOTIFY_CLIENT_ID",
        "Spotify Client ID (from Developer Dashboard)",
        env_vars.get("SPOTIFY_CLIENT_ID", ""),
        r"^[a-f0-9]{32}$"
    )
    if env_vars.get("SPOTIFY_CLIENT_ID"):
        env_vars["SPOTIFY_CLIENT_SECRET"] = prompt_key(
            "SPOTIFY_CLIENT_SECRET",
            "Spotify Client Secret",
            env_vars.get("SPOTIFY_CLIENT_SECRET", ""),
            r"^[a-f0-9]{32}$"
        )
        env_vars["SPOTIFY_REFRESH_TOKEN"] = prompt_key(
            "SPOTIFY_REFRESH_TOKEN",
            "Spotify OAuth Refresh Token",
            env_vars.get("SPOTIFY_REFRESH_TOKEN", "")
        )
        
    # 5. Notion Integration
    env_vars["NOTION_API_KEY"] = prompt_key(
        "NOTION_API_KEY",
        "Notion Internal Integration Token (https://www.notion.so/my-integrations)",
        env_vars.get("NOTION_API_KEY", ""),
        r"^secret_[A-Za-z0-9_-]+$"
    )
    
    # 6. GitHub Integration
    env_vars["GITHUB_TOKEN"] = prompt_key(
        "GITHUB_TOKEN",
        "GitHub Personal Access Token (classic, with repo scope)",
        env_vars.get("GITHUB_TOKEN", ""),
        r"^(ghp|github_pat)_[A-Za-z0-9_]{36,255}$"
    )
    
    save_env_dict(env_vars)
    print("\n====================================================")
    print("Configuration saved successfully to .env!")
    print("You can edit .env directly at any time.")
    print("====================================================")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        sys.exit(0)
