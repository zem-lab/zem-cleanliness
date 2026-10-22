from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from backend.cleanliness_core import html_to_pdf_edge as export


def test_pdf_print_owns_separate_profile_when_desktop_edge_is_running(tmp_path, monkeypatch):
    html = tmp_path / "report.html"
    html.write_text("<html><body>Actual saved report</body></html>", encoding="utf-8")
    desktop_profile = tmp_path / "desktop-profile"
    desktop_profile.mkdir()
    bookmark = desktop_profile / "Bookmarks"
    bookmark.write_text("existing browser data", encoding="utf-8")
    profiles = []
    monkeypatch.setattr(export, "find_pdf_browser_executable", lambda: Path("msedge.exe"))

    def running_desktop_edge(command, **kwargs):
        private = next((a.split("=", 1)[1] for a in command if a.startswith("--user-data-dir=")), None)
        if private is None or "--disable-features=AutoDeElevate" not in command:
            # A shared desktop process or Windows launch handoff can accept
            # the command, exit 0, and produce no PDF.
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert "--no-sandbox" not in command
        profile = Path(private)
        assert profile.is_dir() and profile != desktop_profile
        profiles.append(profile)
        pdf = Path(next(a.split("=", 1)[1] for a in command if a.startswith("--print-to-pdf=")))
        pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 1500 + b"\n%%EOF\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(export.subprocess, "run", running_desktop_edge)
    for name in ("first.pdf", "second.pdf"):
        result = export.html_to_pdf_via_edge(html, tmp_path / name)
        assert result.is_file() and result.stat().st_size > 1000
    assert len(set(profiles)) == 2
    assert all(not p.exists() for p in profiles)
    assert bookmark.read_text(encoding="utf-8") == "existing browser data"


def test_success_exit_without_pdf_is_still_failure_and_cleans_private_profile(tmp_path, monkeypatch):
    html = tmp_path / "report.html"
    html.write_text("<html>Report</html>", encoding="utf-8")
    profiles = []
    monkeypatch.setattr(export, "find_pdf_browser_executable", lambda: Path("msedge.exe"))

    def no_pdf(command, **kwargs):
        profiles.append(Path(next(a.split("=", 1)[1] for a in command if a.startswith("--user-data-dir="))))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(export.subprocess, "run", no_pdf)
    monkeypatch.setattr(export, "_wait_for_complete_pdf", lambda *_: False)
    with pytest.raises(RuntimeError, match="Edge.*exit=0"):
        export.html_to_pdf_via_edge(html, tmp_path / "failed.pdf")
    assert len(profiles) == 2
    assert all(not p.exists() for p in profiles)
    assert not (tmp_path / ".report.edge_print.html").exists()


def test_launcher_exit_zero_waits_for_detached_print_worker(tmp_path, monkeypatch):
    html = tmp_path / "report.html"
    html.write_text("<html>Saved software report</html>", encoding="utf-8")
    pdf = tmp_path / "report.pdf"
    worker_observations = []
    workers = []
    monkeypatch.setattr(export, "find_pdf_browser_executable", lambda: Path("msedge.exe"))

    def launch_then_return(command, **kwargs):
        profile = Path(next(a.split("=", 1)[1] for a in command if a.startswith("--user-data-dir=")))
        pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 1500)

        def finish_printing():
            worker_observations.append((profile.is_dir(), (tmp_path / ".report.edge_print.html").is_file()))
            with pdf.open("ab") as output:
                output.write(b"\n%%EOF\n")

        timer = threading.Timer(0.05, finish_printing)
        workers.append(timer)
        timer.start()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(export.subprocess, "run", launch_then_return)
    try:
        assert export.html_to_pdf_via_edge(html, pdf, timeout_s=2) == pdf
    finally:
        for worker in workers:
            worker.join()
    assert worker_observations == [(True, True)]
    assert pdf.read_bytes().endswith(b"%%EOF\n")
    assert not (tmp_path / ".report.edge_print.html").exists()


@pytest.mark.parametrize("content", [b"x" * 1500, b"%PDF-1.7\n" + b"x" * 1500])
def test_incomplete_or_non_pdf_output_is_not_accepted(tmp_path, content):
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(content)
    assert not export._wait_for_complete_pdf(pdf, 0)


def test_pdf_browser_falls_back_to_edge_when_chrome_is_unavailable(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda _: False)
    monkeypatch.setattr(export, "find_msedge_executable", lambda: Path("available-edge.exe"))
    assert export.find_pdf_browser_executable() == Path("available-edge.exe")
