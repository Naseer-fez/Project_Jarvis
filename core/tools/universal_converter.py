import sys
import subprocess
import csv
import json
from pathlib import Path

# ----------------------------------------------------------------------
# Self-Bootstrapping Dependency Manager
# ----------------------------------------------------------------------
def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"[Universal Converter] Installing missing dependency: {package_name}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
            print(f"[Universal Converter] Successfully installed {package_name}.")
        except Exception as e:
            raise RuntimeError(f"Failed to install package '{package_name}' for conversion: {e}")

# ----------------------------------------------------------------------
# Conversion Implementations
# ----------------------------------------------------------------------
def convert_image(src: Path, dst: Path):
    ensure_package("pillow", "PIL")
    from PIL import Image
    with Image.open(src) as img:
        # Convert RGBA to RGB for formats that do not support transparency (like JPG/BMP)
        if dst.suffix.lower() in [".jpg", ".jpeg", ".bmp"] and img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(dst)

def csv_to_json(src: Path, dst: Path):
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

def json_to_csv(src: Path, dst: Path):
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        if isinstance(data, dict):
            # If it's a single dictionary, wrap it in a list
            data = [data]
        else:
            raise ValueError("JSON data must be an array of objects to convert to CSV.")
    if not data:
        raise ValueError("JSON data is empty.")
        
    headers = list(data[0].keys())
    with open(dst, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def json_to_yaml(src: Path, dst: Path):
    ensure_package("pyyaml", "yaml")
    import yaml
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(dst, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False)

def yaml_to_json(src: Path, dst: Path):
    ensure_package("pyyaml", "yaml")
    import yaml
    with open(src, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def md_to_html(src: Path, dst: Path):
    ensure_package("markdown")
    import markdown
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    html = markdown.markdown(text)
    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Converted Document</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      max-width: 800px;
      margin: 40px auto;
      padding: 0 20px;
      color: #333;
      background-color: #fafafa;
    }}
    h1, h2, h3 {{ color: #111; border-bottom: 1px solid #eaeaea; padding-bottom: 0.3em; }}
    pre {{ background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; font-family: monospace; }}
    code {{ font-family: monospace; background: rgba(27,31,35,0.05); padding: .2em .4em; border-radius: 3px; font-size: 85%; }}
    blockquote {{ border-left: 4px solid #dfe2e5; color: #6a737d; padding-left: 1em; margin-left: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    th, td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
    th {{ background-color: #f6f8fa; }}
  </style>
</head>
<body>
  {html}
</body>
</html>"""
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(full_html)

def txt_to_html(src: Path, dst: Path):
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    import html
    escaped_text = html.escape(text)
    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Plain Text View</title>
</head>
<body style="font-family: monospace; padding: 20px; background: #fafafa; line-height: 1.4;">
  <pre>{escaped_text}</pre>
</body>
</html>"""
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(full_html)

def txt_to_pdf(src: Path, dst: Path):
    ensure_package("fpdf2", "fpdf")
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Replace non-latin1 characters to avoid encoding crashes
            line_clean = line.encode('latin-1', 'replace').decode('latin-1').strip('\n')
            pdf.cell(0, 6, txt=line_clean, ln=1)
            
    pdf.output(str(dst))

def pdf_to_txt(src: Path, dst: Path):
    ensure_package("pypdf")
    from pypdf import PdfReader
    reader = PdfReader(src)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(text)

def data_to_excel(src: Path, dst: Path):
    ensure_package("pandas")
    ensure_package("openpyxl")
    import pandas as pd
    if src.suffix.lower() == ".csv":
        df = pd.read_csv(src)
    elif src.suffix.lower() == ".json":
        df = pd.read_json(src)
    else:
        raise ValueError(f"Cannot convert format {src.suffix} directly to Excel. Must be CSV or JSON.")
    df.to_excel(dst, index=False)

def convert_media_ffmpeg(src: Path, dst: Path):
    try:
        cmd = ["ffmpeg", "-y", "-i", str(src), str(dst)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed or not in system PATH. Audio/Video conversion requires FFmpeg installed.")

# ----------------------------------------------------------------------
# Universal Entry Point
# ----------------------------------------------------------------------
def perform_conversion(source_path: str, target_format: str, output_path: str = None) -> str:
    """
    Main entry point to convert files.
    :param source_path: Path of the input file.
    :param target_format: Target format extension (e.g. 'webp', 'pdf', 'csv', 'json').
    :param output_path: Optional custom output file path.
    :return: Path of the converted file.
    """
    src = Path(source_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")
    if not src.is_file():
        raise ValueError(f"Source path is not a file: {source_path}")
        
    ext_src = src.suffix.lower()
    ext_dst = f".{target_format.lower().lstrip('.')}"
    
    if ext_src == ext_dst:
        raise ValueError(f"Source and target format are the same ({ext_src}). No conversion needed.")
        
    if output_path:
        dst = Path(output_path).resolve()
    else:
        dst = src.with_suffix(ext_dst)
        
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Define route map
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
    audio_video_exts = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".mp4", ".avi", ".mkv", ".webm", ".mov"}
    
    # Conversion Dispatcher
    try:
        # 1. Images
        if ext_src in image_exts and ext_dst in image_exts:
            convert_image(src, dst)
            
        # 2. Data conversions
        elif ext_src == ".csv" and ext_dst == ".json":
            csv_to_json(src, dst)
        elif ext_src == ".json" and ext_dst == ".csv":
            json_to_csv(src, dst)
        elif ext_src == ".json" and ext_dst in (".yaml", ".yml"):
            json_to_yaml(src, dst)
        elif ext_src in (".yaml", ".yml") and ext_dst == ".json":
            yaml_to_json(src, dst)
        elif ext_src in (".csv", ".json") and ext_dst in (".xlsx", ".xls"):
            data_to_excel(src, dst)
            
        # 3. Documents
        elif ext_src == ".md" and ext_dst in (".html", ".htm"):
            md_to_html(src, dst)
        elif ext_src == ".txt" and ext_dst in (".html", ".htm"):
            txt_to_html(src, dst)
        elif ext_src == ".txt" and ext_dst == ".pdf":
            txt_to_pdf(src, dst)
        elif ext_src == ".pdf" and ext_dst == ".txt":
            pdf_to_txt(src, dst)
            
        # 4. Audio/Video (using FFmpeg)
        elif ext_src in audio_video_exts and ext_dst in audio_video_exts:
            convert_media_ffmpeg(src, dst)
            
        # 5. Generic FFmpeg fallback for other extensions if source/destination might be media
        elif ext_src in audio_video_exts or ext_dst in audio_video_exts:
            convert_media_ffmpeg(src, dst)
            
        else:
            # Catch-all: Try using FFmpeg for everything else if it might work
            try:
                convert_media_ffmpeg(src, dst)
            except Exception as ffmpeg_err:
                raise ValueError(
                    f"Unsupported conversion format pair: {ext_src} -> {ext_dst}. "
                    f"FFmpeg fallback also failed: {ffmpeg_err}"
                )
                
        return str(dst)
        
    except Exception as e:
        raise RuntimeError(f"Error converting {src.name} to {ext_dst}: {e}")

# ----------------------------------------------------------------------
# CLI Runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Universal File Converter CLI")
        print("Usage: python universal_converter.py <source_file> <target_format> [output_file]")
        sys.exit(1)
        
    src_file = sys.argv[1]
    tgt_fmt = sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        res = perform_conversion(src_file, tgt_fmt, out_file)
        print(f"Success! Converted file saved to: {res}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
