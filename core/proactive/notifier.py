import logging
import time

logger = logging.getLogger(__name__)


class NotificationManager:
    def notify(self, message: str, level: str = "info", voice_layer=None) -> None:
        ts = time.strftime("%H:%M")
        print(f"\n[{ts}][JARVIS/{level.upper()}] {message}")
        try:
            from plyer import notification

            notification.notify(title="Jarvis", message=message[:256], timeout=5)
        except Exception:
            pass  # plyer optional

        if voice_layer is not None:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(voice_layer.speak(message))
            except Exception:
                pass

    def schedule_reminder(self, message: str, in_seconds: int) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._delayed_notify(message, in_seconds))
        except RuntimeError:
            pass  # no running loop - skip

    async def _delayed_notify(self, message: str, delay: int) -> None:
        import asyncio

        await asyncio.sleep(delay)
        self.notify(message)
