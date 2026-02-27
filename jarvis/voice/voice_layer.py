"""
JARVIS Voice Layer - Session 5
Wake Word: pvporcupine ("Jarvis") — LOCAL, no cloud
STT: openai-whisper (tiny/base) — LOCAL transcription
TTS: Piper TTS — LOCAL voice synthesis
All audio stays ON-DEVICE. Zero cloud dependency.
"""

import asyncio
import logging
import tempfile
import os
import time
import struct
import wave
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.controller_v2 import JarvisController
    from memory.hybrid_memory import HybridMemory

logger = logging.getLogger("JARVIS.VoiceLayer")

# Audio config
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_LENGTH = 512       # Porcupine frame size
SILENCE_THRESHOLD = 500  # RMS threshold for silence detection
SILENCE_DURATION = 1.5   # Seconds of silence before stopping recording
MAX_RECORD_SECONDS = 30  # Safety cap


class VoiceLayer:
    def __init__(self, controller: "JarvisController", memory: "HybridMemory",
                 porcupine_key: str = "", whisper_model: str = "base"):
        self.controller = controller
        self.memory = memory
        self.porcupine_key = porcupine_key
        self.whisper_model_name = whisper_model
        self._whisper = None
        self._porcupine = None
        self._tts_ready = False
        self._running = False
        logger.info(f"VoiceLayer created. Whisper model: {whisper_model}")

    async def initialize(self):
        """Load all voice models — all local, all offline."""
        await asyncio.to_thread(self._load_whisper)
        await asyncio.to_thread(self._load_porcupine)
        await asyncio.to_thread(self._setup_tts)

    def _load_whisper(self):
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self._whisper = whisper.load_model(self.whisper_model_name)
            logger.info("Whisper STT ready.")
        except ImportError:
            logger.error("openai-whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            logger.error(f"Whisper load failed: {e}")

    def _load_porcupine(self):
        try:
            import pvporcupine
            if self.porcupine_key:
                self._porcupine = pvporcupine.create(
                    access_key=self.porcupine_key,
                    keywords=["jarvis"]
                )
                logger.info("Porcupine wake word engine ready. Say 'Jarvis' to activate.")
            else:
                logger.warning(
                    "No Porcupine access key provided. "
                    "Wake word detection disabled. "
                    "Get a FREE key at: https://console.picovoice.ai/"
                )
        except ImportError:
            logger.error("pvporcupine not installed. Run: pip install pvporcupine")
        except Exception as e:
            logger.error(f"Porcupine init failed: {e}")

    def _setup_tts(self):
        """Setup Piper TTS. Falls back to pyttsx3 if Piper not available."""
        try:
            import subprocess
            result = subprocess.run(["piper", "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                self._tts_ready = True
                self._tts_engine = "piper"
                logger.info("Piper TTS ready.")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            import pyttsx3
            self._tts_engine_obj = pyttsx3.init()
            self._tts_engine_obj.setProperty('rate', 175)
            self._tts_ready = True
            self._tts_engine = "pyttsx3"
            logger.info("Fallback TTS (pyttsx3) ready.")
        except ImportError:
            logger.warning("No TTS engine available. Install piper or pyttsx3.")
            self._tts_engine = "none"

    async def speak(self, text: str):
        """Convert text to speech using best available local engine."""
        logger.info(f"TTS: '{text[:80]}...' " if len(text) > 80 else f"TTS: '{text}'")
        print(f"\n🔊 JARVIS: {text}\n")

        if not self._tts_ready:
            return

        await asyncio.to_thread(self._speak_sync, text)

    def _speak_sync(self, text: str):
        try:
            if self._tts_engine == "piper":
                import subprocess
                model_path = Path("models/piper/en_US-lessac-medium.onnx")
                if model_path.exists():
                    proc = subprocess.Popen(
                        ["piper", "--model", str(model_path), "--output-raw"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE
                    )
                    proc.communicate(input=text.encode())
                else:
                    logger.warning(f"Piper model not found at {model_path}. Download from: https://github.com/rhasspy/piper/releases")
            elif self._tts_engine == "pyttsx3":
                self._tts_engine_obj.say(text)
                self._tts_engine_obj.runAndWait()
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text using local Whisper."""
        if not self._whisper:
            logger.error("Whisper not loaded.")
            return ""

        try:
            result = await asyncio.to_thread(
                self._whisper.transcribe, audio_path, fp16=False, language="en"
            )
            transcript = result.get("text", "").strip()
            logger.info(f"Transcript: '{transcript}'")
            return transcript
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    async def record_until_silence(self) -> Optional[str]:
        """Record audio from microphone until silence is detected."""
        try:
            import pyaudio
            import numpy as np
        except ImportError:
            logger.error("pyaudio/numpy not installed. Run: pip install pyaudio numpy")
            return None

        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=SAMPLE_RATE,
            channels=CHANNELS,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1024
        )

        logger.info("Recording... (speak now)")
        print("🎤 Recording... (speak your command)")

        frames = []
        silent_chunks = 0
        required_silent_chunks = int(SILENCE_DURATION * SAMPLE_RATE / 1024)
        max_chunks = int(MAX_RECORD_SECONDS * SAMPLE_RATE / 1024)

        for _ in range(max_chunks):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)

            # RMS-based silence detection
            import array
            shorts = array.array('h', data)
            rms = (sum(s**2 for s in shorts) / len(shorts)) ** 0.5
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks >= required_silent_chunks:
                    break
            else:
                silent_chunks = 0

        stream.stop_stream()
        stream.close()
        pa.terminate()

        if not frames:
            return None

        # Save to temp WAV file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))

        logger.info(f"Audio saved: {tmp.name} ({len(frames)} frames)")
        return tmp.name

    async def run_voice_loop(self):
        """Main voice loop: wake word -> record -> transcribe -> plan -> respond."""
        await self.initialize()

        await self.speak(
            "Jarvis voice system online. "
            + ("Say 'Jarvis' to activate." if self._porcupine else "Voice loop ready. Press Enter to speak.")
        )

        self._running = True

        while self._running:
            try:
                if self._porcupine:
                    detected = await self._listen_for_wake_word()
                    if not detected:
                        continue
                else:
                    # Fallback: press Enter to activate
                    input("\n[Press ENTER to speak to Jarvis]")

                await self.speak("Yes, I'm listening.")

                audio_path = await self.record_until_silence()
                if not audio_path:
                    await self.speak("I didn't catch that. Please try again.")
                    continue

                transcript = await self.transcribe(audio_path)
                
                # Cleanup temp file
                try:
                    os.unlink(audio_path)
                except:
                    pass

                if not transcript or len(transcript) < 2:
                    await self.speak("I couldn't understand that. Could you repeat?")
                    continue

                print(f"\n👤 You said: '{transcript}'")

                # Hand off to controller (this triggers state machine + planner + risk check)
                response = await self.controller.process_input(transcript, source="voice")

                await self.speak(response)

                # Log to semantic memory
                self.memory.log_conversation(
                    user_input=transcript,
                    jarvis_reply=response,
                    source="voice",
                    transcript=transcript,
                    intent=transcript[:100]
                )

            except KeyboardInterrupt:
                logger.info("Voice loop interrupted by user.")
                await self.speak("Shutting down voice system. Goodbye.")
                self._running = False
                break
            except Exception as e:
                logger.error(f"Voice loop error: {e}")
                await asyncio.sleep(1)

    async def _listen_for_wake_word(self) -> bool:
        """Block until 'Jarvis' wake word is detected by Porcupine."""
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(
                rate=self._porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self._porcupine.frame_length
            )

            print("👂 Listening for wake word 'Jarvis'...")
            detected = False

            while not detected:
                pcm = stream.read(self._porcupine.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("h" * self._porcupine.frame_length, pcm)
                keyword_index = self._porcupine.process(pcm_unpacked)
                if keyword_index >= 0:
                    logger.info("Wake word 'Jarvis' detected!")
                    detected = True

            stream.stop_stream()
            stream.close()
            pa.terminate()
            return True
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            await asyncio.sleep(2)
            return False

    def stop(self):
        self._running = False
