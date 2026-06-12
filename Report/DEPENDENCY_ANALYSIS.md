# Dependency Analysis

### INTEGRATIONS-004 (Medium)
**Files:** `integrations/clients/calendar.py`
**Description:** The `calendar` integration returns `True` unconditionally in its `is_available()` method, but its `_list_events` utility relies heavily on the third-party libraries `icalendar` and `dateutil`. If these libraries are missing, the integration registers successfully but throws runtime exceptions.

### LOGS-001 (High)
**Files:** d:\AI\Jarvis\logs\app.log
**Description:** The application logs indicate missing dependencies for critical voice components: `faster-whisper` (STT), `piper-tts` (TTS), and `openwakeword` (Wake Word). The application is forced to fallback to dummy or CLI implementations.

### REQUIREMENTS-001 (Medium)
**Files:** d:\AI\Jarvis\requirements\full.txt
**Description:** Use of the deprecated and unmaintained `fpdf` library.

### REQUIREMENTS-002 (Low)
**Files:** d:\AI\Jarvis\requirements\dev.txt, d:\AI\Jarvis\requirements\base.txt
**Description:** Type stubs for `PyYAML` are included in development requirements, but the actual runtime library `PyYAML` is missing from all requirements files.

