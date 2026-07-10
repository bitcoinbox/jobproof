"""Render a kit's tailored resume markdown to an upload-ready one-page PDF.

Closes the loop from posting -> tailored resume -> a file you can actually submit. Uses a
tiny dependency-free Markdown subset (headings, bullets, bold, inline code, paragraphs) and
a print stylesheet, then drives headless Chrome/Chromium to print it. Best-effort: if no
Chrome is installed it returns None with a clear message rather than failing the kit.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
]

_PRINT_CSS = """
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  color:#182131;background:#fff;font-size:10.6px;line-height:1.32}
.page{padding:.4in .5in}
h1{font-size:21px;line-height:1.1;margin:0 0 4px;border-bottom:2px solid #1f4e79;padding-bottom:7px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.9px;color:#1f4e79;
  border-bottom:1px solid #d9dee7;padding-bottom:2px;margin:11px 0 5px}
h3{font-size:11.4px;margin:7px 0 2px}
p{margin:0 0 5px} ul{margin:3px 0 6px;padding-left:16px} li{margin:1.5px 0}
strong{font-weight:700} code{font-family:ui-monospace,Menlo,monospace;font-size:9.4px;background:#f1f4f8;border-radius:3px;padding:0 3px}
a{color:inherit;text-decoration:none}
@page{size:letter;margin:0}
"""


def _inline(text: str) -> str:
    """Escape, then apply inline **bold**, *italic*, and `code`."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def markdown_to_html(md: str, *, title: str = "Resume") -> str:
    """Convert a resume-markdown subset to a styled, print-ready HTML document."""
    out: list[str] = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in (md or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            close_list()
            continue
        if re.match(r"^#{1,6}\s", line):
            close_list()
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{min(level, 3)}>{_inline(line.lstrip('# ').strip())}</h{min(level, 3)}>")
        elif re.match(r"^[-*]\s+", line):
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = re.sub(r"^[-*]\s+", "", line)
            out.append(f"<li>{_inline(item)}</li>")
        else:
            close_list()
            out.append(f"<p>{_inline(line)}</p>")
    close_list()
    body = "\n".join(out)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_PRINT_CSS}</style></head><body><div class='page'>{body}</div></body></html>"
    )


def find_chrome() -> str | None:
    """Locate a headless-capable Chrome/Chromium binary, or None."""
    for c in _CHROME_CANDIDATES:
        if os.path.isabs(c):
            if os.path.exists(c):
                return c
        elif shutil.which(c):
            return shutil.which(c)
    return None


def html_to_pdf(html_text: str, dest: Path, *, chrome: str | None = None) -> Path | None:
    """Print HTML to a PDF via headless Chrome. Returns the path, or None if Chrome is absent."""
    chrome = chrome or find_chrome()
    if not chrome or not (os.path.isabs(chrome) and os.path.exists(chrome) or shutil.which(chrome)):
        return None
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html_text)
        tmp = f.name
    try:
        subprocess.run(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={dest}",
                f"file://{tmp}",
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    finally:
        os.unlink(tmp)
    return dest if dest.exists() else None


def resume_pdf(resume_markdown: str, dest: Path, *, title: str = "Resume", chrome: str | None = None):
    """Render tailored resume markdown to a one-page PDF at `dest`. Returns the path or None."""
    return html_to_pdf(markdown_to_html(resume_markdown, title=title), dest, chrome=chrome)
