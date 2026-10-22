"""Read-only sample 2 regression check: all 800 particles in desktop-generated reports."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.cleanliness_core.optical_particle_analysis import load_optical_export


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] in {"COMPLETED", "COMPLETED_WITH_FAILURES"}
    source = load_optical_export(Path(manifest["input"]["path"]))
    assert source.sha256 == manifest["input"]["sha256"]
    expected = {p.sequence for p in source.particles}
    selected = manifest["selection"]["sequences"]
    records = manifest["particles"]
    actual = [record["sequence"] for record in records]
    assert len(expected) == len(selected) == len(actual) == 800
    assert expected == set(selected) == set(actual)
    assert all(record["status"] in {"SUCCEEDED", "FAILED"} for record in records)
    successful = [r for r in records if r["status"] == "SUCCEEDED"]
    assert all(r.get("energySpectrum") for r in successful)
    assert not any((r.get("energySpectrum") or {}).get("mock") for r in records)
    root = Path(manifest["taskDirectory"])
    reports = {kind: root / name for kind, name in manifest["reports"].items()}
    assert all(path.is_file() for path in reports.values())
    document = reports["html"].read_text(encoding="utf-8")
    html_sequences = list(map(int, re.findall(r"sequence #(\d+) ·", document)))
    assert len(html_sequences) == 800 and set(html_sequences) == expected
    with reports["csv"].open(encoding="utf-8-sig", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows) == 800
    assert {int(row["sequence"]) for row in csv_rows} == expected
    for record, row in zip(records, csv_rows):
        assert record["status"] == row["status"]
        assert (record.get("error") or "") == row["error"]
    poppler = Path("C:/path/to/MiKTeX/miktex/bin/x64")
    text_path = manifest_path.parent / "full_report_text.txt"
    subprocess.run([str(poppler / "pdftotext.exe"), "-layout", str(reports["pdf"]), str(text_path)], check=True)
    pdf_text = text_path.read_text(encoding="utf-8")
    pdf_sequences = list(map(int, re.findall(r"sequence\s*#(\d+)", pdf_text)))
    assert len(pdf_sequences) == 800 and set(pdf_sequences) == expected
    eds_failed = sum(r["status"] == "FAILED" and bool(r.get("edsAttempts")) for r in records)
    result = {
        "taskId": manifest["taskId"],
        "input": str(source.path),
        "sha256": source.sha256,
        "selected": 800,
        "succeeded": len(successful),
        "localizationFailed": 800 - len(successful) - eds_failed,
        "edsFailed": eds_failed,
        "all800ExactlyOnceInHtmlCsvPdf": True,
        "pdfPages": pdf_text.count("\f"),
        "pdfBytes": reports["pdf"].stat().st_size,
        "reports": {kind: str(path) for kind, path in reports.items()},
    }
    (manifest_path.parent / "final_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), ensure_ascii=False, indent=2))
