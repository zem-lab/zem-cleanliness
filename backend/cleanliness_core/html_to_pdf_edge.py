"""用 Windows Chromium 浏览器将软件原有 HTML 打印为 PDF。"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

_EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)

_SRC_HREF_RE = re.compile(r"""(?P<attr>src|href)=(["'])(?P<url>.*?)\2""", re.IGNORECASE)


def find_msedge_executable() -> Path | None:
    """查找本机 msedge.exe；找不到时返回 None。"""
    for candidate in _EDGE_CANDIDATES:
        if candidate.is_file():
            return candidate
    which = shutil.which("msedge")
    if which:
        return Path(which)
    return None


def find_pdf_browser_executable() -> Path | None:
    """Prefer installed Chrome; retain Edge for machines without Chrome."""
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Google/Chrome/Application/chrome.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return find_msedge_executable()


def _mime_for(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _resolve_local_asset(url: str, html_dir: Path) -> Path | None:
    """把 HTML 中的相对路径 / file URI 解析为本地文件。"""
    value = (url or "").strip()
    if not value or value.startswith(("data:", "http://", "https://", "blob:", "about:")):
        return None
    if value.startswith("file:"):
        parsed = urlparse(value)
        local = Path(unquote(parsed.path))
        # Windows file:///C:/... → path 以 /C:/ 开头
        if os.name == "nt" and str(local).startswith("/") and len(str(local)) > 3 and str(local)[2] == ":":
            local = Path(str(local)[1:])
        return local if local.is_file() else None
    candidate = (html_dir / value.replace("\\", "/")).resolve()
    return candidate if candidate.is_file() else None


def _embed_local_assets_as_data_uri(html_text: str, html_dir: Path) -> str:
    """将本地图片等资源内联为 data URI，避免 Edge headless 本地文件竞态。"""

    def repl(match: re.Match[str]) -> str:
        attr = match.group("attr")
        quote = match.group(2)
        url = match.group("url")
        if url.startswith("data:"):
            return match.group(0)
        # CSS/脚本外链不内联；只处理图片类资源
        if attr.lower() == "href" and not re.search(r"\.(png|jpe?g|gif|webp|svg|bmp)(\?|$)", url, re.I):
            return match.group(0)
        asset = _resolve_local_asset(url, html_dir)
        if asset is None:
            # 相对路径解析失败时，仍尽量改成绝对 file URI（兜底）
            if not url.startswith(("http://", "https://", "file:", "data:")):
                candidate = (html_dir / url.replace("\\", "/")).resolve()
                if candidate.exists():
                    return f"{attr}={quote}{candidate.as_uri()}{quote}"
            return match.group(0)
        raw = asset.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        mime = _mime_for(asset)
        return f"{attr}={quote}data:{mime};base64,{encoded}{quote}"

    return _SRC_HREF_RE.sub(repl, html_text)


def _inject_print_asset_css(html_text: str) -> str:
    """确保打印时背景色/图片可见，并压缩首页标定网格占用高度。"""
    snippet = """<style data-edge-print-fix>
html,body{-webkit-print-color-adjust:exact;print-color-adjust:exact;background:#fff!important}
@page{size:A4 landscape;margin:8mm}
@media print{
  .page{padding:0!important;margin:0!important;box-shadow:none!important}
  .report-header{padding-bottom:6px!important}.brand-logo{width:150px!important}
  .title-block{padding:8px 0 6px!important}h1{font-size:22px!important;margin:2px 0!important}
  h2{margin:10px 0 6px!important;font-size:13px!important}.sub{margin:2px 0 8px!important}
  .metric{padding:6px 8px!important}.metric strong{font-size:16px!important}
  .equation{padding:6px 8px!important;font-size:11px!important}.audit div{padding:5px 8px!important}
  .cal-grid{gap:6px!important;break-inside:avoid;page-break-inside:avoid}
  .cal-pair{padding:4px!important;break-inside:avoid;page-break-inside:avoid}
  .cal-pair img{
    display:block!important;width:100%!important;height:88px!important;min-height:88px!important;
    object-fit:contain!important;background:#101517!important;
    -webkit-print-color-adjust:exact;print-color-adjust:exact
  }
  img{image-rendering:auto;-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
</style>"""
    if "data-edge-print-fix" in html_text:
        return html_text
    if re.search(r"</head\s*>", html_text, re.I):
        return re.sub(r"</head\s*>", snippet + "</head>", html_text, count=1, flags=re.I)
    return snippet + html_text


def prepare_html_for_edge_print(html_path: Path) -> Path:
    """生成仅用于 Edge 打印的临时 HTML（内联本地图片；不改正式 HTML）。"""
    html_path = Path(html_path).resolve()
    html_dir = html_path.parent
    text = html_path.read_text(encoding="utf-8")
    text = _embed_local_assets_as_data_uri(text, html_dir)
    text = _inject_print_asset_css(text)
    # 放在同目录，相对路径兜底时仍可解析
    temp_path = html_dir / f".{html_path.stem}.edge_print{html_path.suffix}"
    temp_path.write_text(text, encoding="utf-8")
    return temp_path


def _wait_for_complete_pdf(pdf_path: Path, timeout_s: float) -> bool:
    """Windows browser launchers can return before their print worker finishes."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            with pdf_path.open("rb") as file:
                file.seek(0, 2)
                size = file.tell()
                if size > 1000:
                    file.seek(0)
                    header = file.read(5)
                    file.seek(max(0, size - 2048))
                    if header == b"%PDF-" and b"%%EOF" in file.read():
                        return True
        except OSError:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))


def html_to_pdf_via_edge(
    html_path: Path,
    pdf_path: Path,
    *,
    timeout_s: float = 180.0,
) -> Path:
    """通过已安装的 Chromium 浏览器将 HTML 打印为 PDF。

    成功条件：PDF 大于 1000 字节，且具有完整头尾标记。
    会生成临时打印 HTML（data URI 内联本地图），打印后删除。
    """
    edge = find_pdf_browser_executable()
    if edge is None:
        raise RuntimeError("未找到 Chrome 或 Edge，无法将清洁度报告 HTML 打印为 PDF")

    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    if not html_path.is_file():
        raise RuntimeError(f"Edge 打印失败：HTML 不存在：{html_path}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except OSError:
            pass

    print_html: Path | None = None
    browser_profile: tempfile.TemporaryDirectory | None = None
    try:
        print_html = prepare_html_for_edge_print(html_path)
        html_uri = print_html.as_uri()
        # A running desktop Edge can consume this command through its shared
        # profile and exit successfully without printing. Give each export its
        # own profile so the headless process owns the print operation.
        browser_profile = tempfile.TemporaryDirectory(
            prefix="zem-cleanliness-pdf-", ignore_cleanup_errors=True
        )
        # allow-file-access-from-files：允许 file 页访问本地资源（data URI 为主，此为兜底）
        # virtual-time-budget：给解码/排版留时间
        common = [
            f"--user-data-dir={browser_profile.name}",
            "--no-first-run",
            "--no-default-browser-check",
            # Match Chromium automation launchers on Windows: automatic
            # de-elevation can hand off and exit 0 without executing headless
            # printing. This leaves the browser's renderer sandbox enabled.
            "--disable-features=AutoDeElevate",
            f"--print-to-pdf={pdf_path}",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--allow-file-access-from-files",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=60000",
            html_uri,
        ]
        attempts = (
            ["--headless=new", *common],
            ["--headless", *common],
        )
        last_error = ""
        cwd = str(html_path.parent)
        env = os.environ.copy()

        for args in attempts:
            command = [str(edge), *args]
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                last_error = f"超时（>{timeout_s:.0f}s）"
                raise RuntimeError(f"Edge 打印失败：{last_error}") from exc
            except OSError as exc:
                last_error = str(exc)
                continue

            # Exit 0 can mean only that Edge's launcher successfully handed off
            # printing to another process. Keep the HTML and private profile
            # alive until the output is complete, within the original timeout.
            remaining = max(0.0, timeout_s - (time.monotonic() - started))
            if _wait_for_complete_pdf(pdf_path, remaining if completed.returncode == 0 else 0):
                return pdf_path

            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            last_error = stderr or stdout or f"exit={completed.returncode}"

        size = pdf_path.stat().st_size if pdf_path.is_file() else 0
        detail = last_error or f"PDF 无效（size={size}）"
        raise RuntimeError(f"Edge 打印失败：{detail}")
    finally:
        if browser_profile is not None:
            browser_profile.cleanup()
        if print_html is not None:
            try:
                print_html.unlink(missing_ok=True)
            except OSError:
                pass
