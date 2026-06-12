import asyncio
import sys
import pyautogui
from core.tools.gui_control import move_mouse, type_text

async def run_test():
    print("Starting visual GUI test...")
    # Move mouse to different corners
    print("Moving mouse to (100, 100)")
    await move_mouse(100, 100, duration=1.0)
    await asyncio.sleep(1)
    
    print("Moving mouse to (500, 500)")
    await move_mouse(500, 500, duration=1.0)
    await asyncio.sleep(1)
    
    # Open notepad via pyautogui
    print("Opening Notepad...")
    pyautogui.hotkey('win', 'r')
    await asyncio.sleep(1)
    pyautogui.typewrite('notepad\n', interval=0.05)
    await asyncio.sleep(2)
    
    # Type text using Jarvis gui_control tool
    text = "Hello! Jarvis here. I am typing on the screen visually and moving the mouse."
    print(f"Typing text: {text}")
    await type_text(text, interval=0.05)
    print("Test complete.")

if __name__ == "__main__":
    asyncio.run(run_test())
