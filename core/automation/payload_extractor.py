from pathlib import Path
import re


_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".json", ".yaml", ".yml",
    ".csv", ".tsv", ".py", ".js", ".ts", ".html",
    ".css", ".ini", ".log",
}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def read_text_file(path: Path, max_bytes: int = 2_000_000) -> str:
    data = path.read_bytes()[: max(1, max_bytes)]
    return data.decode("utf-8", errors="replace")


class PayloadExtractor:
    def __init__(self, max_text_chars_per_item: int, video_frame_interval_seconds: float, max_video_samples: int):
        self.max_text_chars_per_item = max_text_chars_per_item
        self.video_frame_interval_seconds = video_frame_interval_seconds
        self.max_video_samples = max_video_samples

    def extract_text_payload(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in _TEXT_EXTENSIONS:
            raw = read_text_file(path)
            return _truncate(raw, self.max_text_chars_per_item)
        if suffix in _IMAGE_EXTENSIONS:
            text = self.extract_text_from_image(path)
            return _truncate(text, self.max_text_chars_per_item)
        if suffix in _VIDEO_EXTENSIONS:
            text = self.extract_text_from_video(path)
            return _truncate(text, self.max_text_chars_per_item)
        return f"Unsupported file type for direct parsing: {path.name}"

    def extract_text_from_image(self, path: Path) -> str:
        try:
            from PIL import Image
            import pytesseract
        except Exception as exc:  # noqa: BLE001
            return f"OCR dependency missing for image '{path.name}': {exc}"

        try:
            with Image.open(path) as image:
                raw = pytesseract.image_to_string(image)
        except Exception as exc:  # noqa: BLE001
            return f"Image OCR failed for '{path.name}': {exc}"

        text = _normalize_text(raw)
        if not text:
            return f"No OCR text found in image '{path.name}'."
        return text

    def extract_text_from_video(self, path: Path) -> str:
        try:
            import cv2  # type: ignore[import]
            from PIL import Image
            import pytesseract
        except Exception as exc:  # noqa: BLE001
            return f"Video OCR dependency missing for '{path.name}': {exc}"

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return f"Could not open video '{path.name}'."

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 1.0
        sample_every_frames = max(1, int(round(fps * self.video_frame_interval_seconds)))

        frame_index = 0
        captured = 0
        snippets: list[str] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        try:
            while captured < self.max_video_samples:
                if total_frames > 0 and frame_index >= total_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    break
                    
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    raw = pytesseract.image_to_string(image)
                    text = _normalize_text(raw)
                    if text:
                        second = frame_index / max(fps, 1.0)
                        snippets.append(f"[t={second:.1f}s] {text}")
                        captured += 1
                except Exception:
                    pass
                frame_index += sample_every_frames
        finally:
            cap.release()

        if not snippets:
            return f"No OCR text found in video '{path.name}'."
        return "\n".join(snippets)
