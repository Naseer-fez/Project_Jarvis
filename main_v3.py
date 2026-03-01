"""
main_v3.py
----------
JARVIS Phase 5 — Voice Interaction Loop

Flow:
  1. Load config from config/jarvis_config.ini
  2. Initialize all subsystems (Memory, LLM, STT, TTS, WakeWord, Serial)
  3. Listen for wake word in background thread
  4. On wake: record audio -> STT -> intent classify -> dispatch -> TTS response
  5. Repeat forever; Ctrl+C to exit
"""

import configparser
import logging
import signal
import sys
import threading
from pathlib import Path


def _setup_logging(config: configparser.ConfigParser):
    level_str = config.get("LOGGING", "level", fallback="INFO")
    log_file = config.get("LOGGING", "log_file", fallback=None)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level_str.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def _load_config(path: str = "config/jarvis_config.ini") -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Create it from config/jarvis_config.ini in the project."
        )
    config.read(config_path, encoding="utf-8")
    return config


class JarvisVoiceLoop:
    """Orchestrates the full voice interaction pipeline."""

    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.logger = logging.getLogger("JarvisMain")

        self._wake_event = threading.Event()
        self._shutdown_event = threading.Event()

        self.memory = None
        self.controller = None
        self.classifier = None
        self.dispatcher = None
        self.stt = None
        self.tts = None
        self.wake_detector = None
        self.serial = None

    def _init_memory_and_brain(self):
        """Load Phase 4 HybridMemory + LLM controller."""
        self.logger.info("Initializing memory and LLM brain...")
        try:
            # FIXED IMPORT PATHS HERE
            from core.memory.hybrid_memory import HybridMemory
            from core.controller_v2 import Controller
            from core.planning.intents import IntentClassifier

            self.memory = HybridMemory(self.config)
            self.controller = Controller(self.config)
            self.classifier = IntentClassifier(self.config)
            self.logger.info("Memory and brain initialized.")
        except ImportError as e:
            self.logger.error(
                "Could not import Phase 4 modules: %s\n"
                "Ensure hybrid_memory.py, controller.py, classifier.py exist.", e
            )
            raise

    def _init_voice(self):
        """Initialize STT and TTS subsystems."""
        self.logger.info("Initializing voice subsystems...")

        stt_config = {
            "model_size":                self.config.get("STT", "model_size", fallback="base.en"),
            "whisper_cache_dir":         self.config.get("PATHS", "whisper_cache"),
            "language":                  self.config.get("STT", "language", fallback="en"),
            "record_timeout_seconds":    self.config.get("STT", "record_timeout_seconds", fallback="8"),
            "silence_threshold_seconds": self.config.get("STT", "silence_threshold_seconds", fallback="1.5"),
            "energy_multiplier":         self.config.get("STT", "energy_multiplier", fallback="2.5"),
            "sample_rate":               self.config.get("STT", "sample_rate", fallback="16000"),
            "channels":                  self.config.get("STT", "channels", fallback="1"),
            "microphone_index":          self.config.get("WAKE_WORD", "microphone_index", fallback="-1"),
        }

        tts_config = {
            "piper_model_dir":    self.config.get("PATHS", "piper_model_dir"),
            "voice_model":        self.config.get("TTS", "voice_model", fallback="en_US-lessac-medium"),
            "speaker_id":         self.config.get("TTS", "speaker_id", fallback="0"),
            "output_device_index": self.config.get("TTS", "output_device_index", fallback="-1"),
        }

        from core.voice.stt import SpeechToText
        from core.voice.tts import TextToSpeech

        self.stt = SpeechToText(stt_config)
        self.tts = TextToSpeech(tts_config)
        self.logger.info("Voice subsystems ready.")

    def _init_wake_word(self):
        """Initialize Porcupine wake word detector."""
        self.logger.info("Initializing wake word detector...")

        wake_config = {
            "access_key":       self.config.get("WAKE_WORD", "access_key"),
            "keyword_path":     self.config.get("PATHS", "porcupine_keyword_path"),
            "sensitivity":      self.config.get("WAKE_WORD", "sensitivity", fallback="0.5"),
            "microphone_index": self.config.get("WAKE_WORD", "microphone_index", fallback="-1"),
        }

        from core.voice.wake_word import WakeWordDetector
        self.wake_detector = WakeWordDetector(wake_config)
        self.logger.info("Wake word detector ready.")

    def _init_hardware(self):
        """Initialize serial controller (with graceful fallback)."""
        self.logger.info("Initializing hardware serial controller...")

        hw_config = {
            "com_port":         self.config.get("HARDWARE", "com_port", fallback="COM7"),
            "baud_rate":        self.config.get("HARDWARE", "baud_rate", fallback="115200"),
            "timeout_seconds":  self.config.get("HARDWARE", "timeout_seconds", fallback="2"),
            "require_hardware": self.config.get("HARDWARE", "require_hardware", fallback="false"),
        }

        from core.hardware.serial_controller import SerialController
        self.serial = SerialController(hw_config)

        if self.serial.is_connected():
            self.logger.info("Hardware connected on %s.", hw_config["com_port"])
        else:
            self.logger.info("Hardware running in simulation mode.")

    def _init_dispatcher(self):
        """Wire up the dispatcher with all subsystems."""
        from core.execution.dispatcher import Dispatcher
        self.dispatcher = Dispatcher(
            controller=self.controller,
            memory=self.memory,
            serial_controller=self.serial,
        )

    def _on_wake_word(self):
        """Callback from wake word thread — signals main loop."""
        self._wake_event.set()

    def _process_voice_turn(self):
        """
        Single voice interaction turn:
          record -> transcribe -> classify -> dispatch -> speak
        """
        # 1. Acknowledge wake word
        self.tts.async_speak("Yes?")

        # 2. Record and transcribe
        self.logger.info("[LOOP] Recording speech...")
        print("\n🎙️  Listening...")
        transcript = self.stt.transcribe()

        if not transcript:
            self.logger.info("No speech detected — returning to wake word standby.")
            self.tts.async_speak("I didn't catch that.")
            return

        print(f"📝 You said: {transcript}")
        self.logger.info("[LOOP] Transcript: '%s'", transcript)

        # 3. Classify intent
        self.logger.info("[LOOP] Classifying intent...")
        print("🧠 Thinking...")
        intent, entities = self.classifier.classify(transcript)
        self.logger.info("[LOOP] Intent='%s' Entities=%s", intent, entities)

        # 4. Store in short-term memory
        self.memory.add_to_session({"role": "user", "content": transcript})

        # 5. Dispatch and get response
        response = self.dispatcher.dispatch(intent, transcript, entities)
        self.logger.info("[LOOP] Response: '%s'", response[:100])

        # 6. Store response in memory
        self.memory.add_to_session({"role": "assistant", "content": response})

        # 7. Speak response
        print(f"🤖 JARVIS: {response}")
        self.tts.wait_until_done()  # finish "Yes?" before main response
        self.tts.speak(response)

    def run(self):
        """Main entry point — initialize everything and start the voice loop."""
        self.logger.info("=" * 60)
        self.logger.info("  JARVIS v3 — Voice Layer Starting")
        self.logger.info("=" * 60)

        try:
            self._init_memory_and_brain()
            self._init_voice()
            self._init_hardware()
            self._init_dispatcher()
            self._init_wake_word()
        except Exception as e:
            self.logger.critical("Initialization failed: %s", e, exc_info=True)
            sys.exit(1)

        def _signal_handler(sig, frame):
            print("\n\n Shutting down JARVIS...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        self.wake_detector.start(callback=self._on_wake_word)

        print("\n JARVIS is online. Say 'Jarvis' to activate.\n")
        self.tts.speak("JARVIS online and ready.")

        # ── Main loop ────────────────────────────────────────────────────────
        while not self._shutdown_event.is_set():
            woke = self._wake_event.wait(timeout=1.0)

            if self._shutdown_event.is_set():
                break

            if woke:
                self._wake_event.clear()
                try:
                    self._process_voice_turn()
                except Exception as e:
                    self.logger.error("Error during voice turn: %s", e, exc_info=True)
                    try:
                        self.tts.speak("Something went wrong. Please try again.")
                    except Exception:
                        pass

                self.logger.info("[LOOP] Returning to wake word standby.")
                print("\n Waiting for wake word...\n")

        # ── Cleanup ──────────────────────────────────────────────────────────
        self.logger.info("Cleaning up resources...")
        if self.wake_detector:
            self.wake_detector.stop()
        if self.serial:
            self.serial.close()
        if self.tts:
            self.tts.stop()
        self.logger.info("JARVIS shut down cleanly.")
        print("JARVIS offline.")


if __name__ == "__main__":
    config = _load_config("config/jarvis_config.ini")
    _setup_logging(config)
    jarvis = JarvisVoiceLoop(config)
    jarvis.run()