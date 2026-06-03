from unittest.mock import AsyncMock

async def controlled_sleep(delay: float):
    """
    A controlled sleep function to replace asyncio.sleep in tests.
    This doesn't actually sleep, but returns immediately, avoiding test delays.
    """
    pass

class ControlledAsyncMock(AsyncMock):
    """
    An AsyncMock that integrates cleanly without wall-clock delays.
    """
    pass
