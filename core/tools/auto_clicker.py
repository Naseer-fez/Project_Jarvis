"""Desktop Auto-Clicker utility for continuously finding and clicking dynamic UI elements."""

import argparse
import asyncio
import logging
import sys

from core.tools.gui_control import click_screen_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoClicker")


async def run_auto_clicker(target: str, interval: float, continuous: bool, min_confidence: float) -> None:
    """Run the auto clicker loop."""
    logger.info("Starting Auto-Clicker for target: '%s'", target)
    logger.info("Interval: %.1f seconds | Continuous: %s", interval, continuous)
    
    attempts = 0
    while True:
        attempts += 1
        logger.debug("Attempt %d: Searching for target...", attempts)
        try:
            # We use click_screen_target which combines OCR and Vision
            result = await click_screen_target(
                target=target,
                occurrence=1,
                button="left",
                match_mode="contains",
                min_confidence=min_confidence,
            )
            
            if result.success:
                logger.info("Successfully clicked target! Result: %s", result.data)
                if not continuous:
                    logger.info("Continuous mode is disabled. Exiting after successful click.")
                    break
            else:
                logger.debug("Target not found or click failed: %s", result.error)
                
        except Exception as e:
            logger.error("Error during auto-clicker loop: %s", e, exc_info=True)
            
        logger.debug("Waiting %.1f seconds before next check...", interval)
        await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop Auto-Clicker using Vision and OCR.")
    parser.add_argument("-t", "--target", type=str, required=True, help="Description or text of the target to click.")
    parser.add_argument("-i", "--interval", type=float, default=5.0, help="Seconds to wait between checks.")
    parser.add_argument("-c", "--continuous", action="store_true", help="Keep running even after successfully clicking.")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Minimum confidence threshold for Vision matching.")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_auto_clicker(
            target=args.target,
            interval=args.interval,
            continuous=args.continuous,
            min_confidence=args.min_confidence,
        ))
    except KeyboardInterrupt:
        logger.info("Auto-Clicker stopped by user (Ctrl+C).")
        sys.exit(0)


if __name__ == "__main__":
    main()
