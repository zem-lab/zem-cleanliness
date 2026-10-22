import base64
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from backend.cleanliness_core.cleanliness_report import (
    _element_columns,
    _exclude_element_set,
    _particle_element_columns,
    _sem_evidence_image_key,
    _sem_marker_legend,
    build_cleanliness_reports,
    summarize_cleanliness,
    write_cleanliness_csv,
    write_cleanliness_html,
    write_cleanliness_pdf,
)
from backend.cleanliness_core.html_to_pdf_edge import find_pdf_browser_executable, html_to_pdf_via_edge


def _particle_with_elements(elements: list[tuple[str, float]], *, status: str = "SUCCEEDED") -> dict:
    return {
        "sequence": 1,
        "particleId": "p1",
        "status": status,
        "label": "metal",
        "lengthUm": 10.0,
        "widthUm": 5.0,
        "areaUm2": 40.0,
        "roundness": 0.8,
        "aspect": 2.0,
        "equivalentDiameterUm": 7.0,
        "images": {},
        "localization": {"centerErrorUm": 1.0},
        "energySpectrum": {
            "device": "fake",
            "elements": [{"symbol": symbol, "value": value} for symbol, value in elements],
        },
    }


def _minimal_manifest(particles: list[dict]) -> dict:
    return {
        "taskId": "element-filter-test",
        "status": "COMPLETED",
        "input": {"particleCount": len(particles)},
        "selection": {"count": len(particles), "filter": {}},
        "calibration": {
            "result": {
                "equation": {"stageX": "x' = x", "stageY": "y' = y"},
                "fit": {"rmsErrorUm": 1.0},
                "validation": {"meanErrorUm": 1.0, "maxErrorUm": 2.0},
            },
            "anchors": [],
            "validation": [],
        },
        "particles": particles,
    }


def test_report_uses_exact_source_mask_and_separates_coordinate_systems(tmp_path):
    source = Path(__file__).parent / "fixtures/mask_matching_sample2/optical.json"
    optical = json.loads(source.read_text(encoding="utf-8"))["particles"][0]
    record = _particle_with_elements([])
    record.update(sequence=optical["sequence"], particleId=optical["particleId"],
                  opticalAbPositionUm={"x":0,"y":-123.4567}, targetStageUm={"x":-1000,"y":2000},
                  actualStageUm={"x":-999.75,"y":2000.25},
                  localization={"observedStageUm":{"x":-900,"y":2100}})
    manifest = _minimal_manifest([record])
    manifest["input"].update(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    before = copy.deepcopy(manifest)
    document = write_cleanliness_html(tmp_path,manifest).read_text(encoding="utf-8")
    assert optical["maskDataBase64"].split(",")[-1] in document
    assert 'alt="原始光镜掩码"' in document
    assert f"X {optical['position']['x']:.3f}" in document
    assert 'X 0.000　Y -123.457 μm' in document
    assert 'X -1000.000　Y 2000.000 μm' in document
    assert 'X -999.750　Y 2000.250 μm' in document
    assert 'X -900.000　Y 2100.000 μm' in document
    assert manifest == before


def test_report_does_not_claim_failed_candidate_as_confirmed_particle_position(tmp_path):
    record = _particle_with_elements([],status="FAILED")
    record.update(targetStageUm={"x":10,"y":20},actualStageUm={"x":10.25,"y":20.25},
                  localization={"observedStageUm":{"x":999,"y":888}},match={"status":"FAILED_MASK_IDENTITY"})
    document=write_cleanliness_html(tmp_path,_minimal_manifest([record])).read_text(encoding="utf-8")
    assert '电镜颗粒位置（匹配估计）</span><strong>未确认' in document
    assert 'X 999.000' not in document
    assert 'X 10.250　Y 20.250 μm' in document
    assert '原始光镜掩码不可用' in document


def test_report_rejects_changed_optical_source_fingerprint(tmp_path):
    source=Path(__file__).parent/'fixtures/mask_matching_sample2/optical.json'
    optical=json.loads(source.read_text(encoding='utf-8'))['particles'][0]
    record=_particle_with_elements([])
    record.update(sequence=optical['sequence'],particleId=optical['particleId'])
    manifest=_minimal_manifest([record]);manifest['input'].update(path=str(source),sha256='changed')
    document=write_cleanliness_html(tmp_path,manifest).read_text(encoding='utf-8')
    assert '指纹与任务不一致' in document
    assert 'alt="原始光镜掩码"' not in document


def test_coordinate_option_hides_coordinates_but_keeps_original_mask(tmp_path):
    source=Path(__file__).parent/'fixtures/mask_matching_sample2/optical.json'
    optical=json.loads(source.read_text(encoding='utf-8'))['particles'][0]
    record=_particle_with_elements([])
    record.update(sequence=optical['sequence'],particleId=optical['particleId'],targetStageUm={'x':12345.678,'y':9})
    manifest=_minimal_manifest([record]);manifest['input'].update(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    shown=write_cleanliness_html(tmp_path,manifest).read_text(encoding='utf-8')
    hidden=write_cleanliness_html(tmp_path,manifest,{'showCoordinates':False}).read_text(encoding='utf-8')
    assert '电镜预测台坐标' in shown and '12345.678' in shown
    assert '电镜预测台坐标' not in hidden and '12345.678' not in hidden
    assert optical['maskDataBase64'].split(',')[-1] in hidden
    assert 'alt="原始光镜掩码"' in hidden


def test_full_report_keeps_all_800_particles_and_failure_reasons(tmp_path: Path):
    particles = []
    for sequence in range(1, 801):
        particle = _particle_with_elements([])
        particle.update(sequence=sequence, particleId=f"p{sequence}")
        if sequence % 2 == 0:
            particle.update(status="FAILED", error="no match <threshold>", energySpectrum=None)
        if sequence % 4 == 0:
            particle["edsAttempts"] = [{"status": "FAILED"}]
            particle["error"] = "Oxford no connect"
        particles.append(particle)
    manifest = _minimal_manifest(particles)
    original = copy.deepcopy(manifest)
    document = write_cleanliness_html(tmp_path, manifest).read_text(encoding="utf-8")
    assert document.count('<section class="particle">') == 800
    assert document.count('检测状态：成功') == 400
    assert document.count('检测状态：失败') == 400
    assert 'no match &lt;threshold&gt;' in document
    assert '失败阶段：EDS' in document
    assert '失败阶段：SEM 定位/采集' in document
    assert 'src=""' not in document
    for sequence in range(1, 801):
        assert document.count(f'sequence #{sequence} ·') == 1
    with write_cleanliness_csv(tmp_path, manifest).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 800
    assert rows[1]['error'] == 'no match <threshold>'
    assert rows[3]['failureStage'] == 'EDS'
    assert manifest == original


def test_report_discloses_reused_calibration_and_explicit_selection(tmp_path: Path):
    manifest = _minimal_manifest([_particle_with_elements([])])
    manifest['input']['particleCount'] = 800
    manifest['selection']['sequences'] = [1]
    manifest['selection']['filter'] = {'minLengthUm': None, 'labels': [], 'maxParticles': None}
    manifest['calibrationReuse'] = {'sourceTaskId': 'source<&>'}
    document = write_cleanliness_html(tmp_path, manifest).read_text(encoding="utf-8")
    assert '沿用已确认标定' in document
    assert 'source&lt;&amp;&gt;' in document
    assert '本次重新标定' not in document
    assert '按指定序号选择 1 个颗粒' in document
    assert '执行全部有效颗粒' not in document


def test_pending_particles_are_not_counted_as_failures():
    records = [_particle_with_elements([], status=status) for status in ['SUCCEEDED', 'FAILED', 'RUNNING']]
    manifest = _minimal_manifest(records)
    manifest['selection']['count'] = 800
    summary = summarize_cleanliness(manifest)
    assert summary['processed'] == 2
    assert summary['failed'] == 1
    assert summary['pending'] == 798


def test_failed_particle_includes_capture_without_claiming_match(tmp_path: Path):
    record = _particle_with_elements([], status='FAILED')
    record['captureAttempts'] = [{'displayPath': 'sem/attempt1.png'}, {'error': 'capture error'}]
    document = write_cleanliness_html(tmp_path, _minimal_manifest([record])).read_text(encoding='utf-8')
    assert 'src="sem/attempt1.png"' in document
    assert '采集视场，未确认匹配' in document


def test_sem_evidence_image_prefers_annotated():
    assert (
        _sem_evidence_image_key(
            {
                "images": {
                    "semAnnotatedImage": "analysis_annotated/0001.png",
                    "semParticleImage": "analysis_particles/0001.png",
                }
            }
        )
        == "semAnnotatedImage"
    )


def test_sem_evidence_image_falls_back_to_particle():
    assert (
        _sem_evidence_image_key(
            {"images": {"semParticleImage": "analysis_particles/0001.png"}}
        )
        == "semParticleImage"
    )


def test_sem_marker_legend_covers_cross_meanings():
    meanings = " ".join(item["meaning"] for item in _sem_marker_legend())
    assert "视场中心" in meanings
    assert "颗粒中心" in meanings
    assert "EDS" in meanings


def test_html_print_css_is_a4_landscape(tmp_path: Path):
    html = write_cleanliness_html(
        tmp_path,
        {
            "taskId": "css-check",
            "status": "COMPLETED",
            "input": {"particleCount": 0},
            "selection": {"count": 0, "filter": {}},
            "calibration": {},
            "particles": [],
        },
    )
    text = html.read_text(encoding="utf-8")
    assert "size:A4 landscape" in text
    assert "@media print" in text
    assert "break-after:page" in text


def test_exclude_elements_default_is_o_and_c():
    assert _exclude_element_set(None) == {"O", "C"}
    assert _exclude_element_set({}) == {"O", "C"}
    assert _exclude_element_set({"elementColumns": 5}) == {"O", "C"}
    assert _exclude_element_set({"excludeElements": []}) == set()
    assert _exclude_element_set({"excludeElements": ["o", "c", "n"]}) == {"O", "C", "N"}


def test_element_columns_excludes_o_c_by_default():
    records = [
        _particle_with_elements([("O", 40.0), ("C", 30.0), ("Fe", 20.0), ("Cr", 10.0)])
    ]
    columns = _element_columns(records, maximum=5, exclude=_exclude_element_set(None))
    assert "O" not in columns
    assert "C" not in columns
    assert "Fe" in columns


def test_element_columns_keeps_o_c_when_exclude_empty():
    records = [
        _particle_with_elements([("O", 40.0), ("C", 30.0), ("Fe", 20.0)])
    ]
    columns = _element_columns(
        records,
        maximum=5,
        exclude=_exclude_element_set({"excludeElements": []}),
    )
    assert "O" in columns
    assert "C" in columns
    assert "Fe" in columns


def test_particle_element_columns_rank_by_value_and_drop_oc():
    record = _particle_with_elements(
        [("O", 56.56), ("C", 0.0), ("Al", 63.59), ("Ag", 2.58), ("Fe", 0.0)]
    )
    columns = _particle_element_columns(
        record, maximum=5, exclude=_exclude_element_set(None)
    )
    assert columns == ["Al", "Ag"]


def test_particle_element_columns_keeps_only_top_n():
    record = _particle_with_elements(
        [("Fe", 70.0), ("Cr", 10.0), ("Si", 5.0), ("Ca", 3.0), ("Mn", 2.0), ("Zn", 1.0)]
    )
    columns = _particle_element_columns(
        record, maximum=5, exclude=_exclude_element_set(None)
    )
    assert columns == ["Fe", "Cr", "Si", "Ca", "Mn"]
    assert "Zn" not in columns


def test_html_uses_per_particle_top5_and_pads_empty_columns(tmp_path: Path):
    aluminum = _particle_with_elements(
        [("O", 56.56), ("C", 0.0), ("Al", 63.59), ("Ag", 2.58)]
    )
    aluminum["particleId"] = "al-particle"
    steel = _particle_with_elements(
        [("Fe", 70.0), ("Cr", 10.0), ("Si", 5.0), ("Ca", 3.0), ("Mn", 2.0), ("Zn", 1.0)]
    )
    steel["particleId"] = "fe-particle"
    steel["sequence"] = 2
    html_text = write_cleanliness_html(
        tmp_path, _minimal_manifest([aluminum, steel])
    ).read_text(encoding="utf-8")
    sections = html_text.split('<section class="particle">')[1:]
    assert len(sections) == 2

    assert "<th>Al</th>" in sections[0]
    assert "<th>Ag</th>" in sections[0]
    assert "<th>Fe</th>" not in sections[0]
    assert sections[0].count("<th></th>") == 3
    # 剔除 O/C 后归一化：Al 63.59/66.17 → 96.101，Ag 2.58/66.17 → 3.899
    assert "<td>96.101</td>" in sections[0]
    assert "<td>3.899</td>" in sections[0]
    assert "<th>O</th>" not in sections[0]
    assert "<th>C</th>" not in sections[0]

    assert "<th>Fe</th>" in sections[1]
    assert "<th>Cr</th>" in sections[1]
    assert "<th>Zn</th>" not in sections[1]
    assert "<th>Al</th>" not in sections[1]
    # 6 个元素超出 5 列：Mn、Zn 并入「其他」
    assert "<th>Mn</th>" not in sections[1]
    assert "<th>其他</th>" in sections[1]
    assert sections[1].count("<th></th>") == 0


def test_html_overflow_elements_merged_into_other_column(tmp_path: Path):
    particle = _particle_with_elements(
        [("Fe", 40.0), ("Cr", 20.0), ("Ni", 15.0), ("Mn", 10.0), ("Si", 8.0), ("Ca", 7.0)]
    )
    html_text = write_cleanliness_html(
        tmp_path, _minimal_manifest([particle])
    ).read_text(encoding="utf-8")

    # 前 4 个元素 + 「其他」（Si 8 + Ca 7 = 15），合计仍为 100
    assert "<th>Fe</th>" in html_text
    assert "<th>Mn</th>" in html_text
    assert "<th>其他</th>" in html_text
    assert "<td>15.000</td>" in html_text
    assert "<th>Si</th>" not in html_text
    assert "<th>Ca</th>" not in html_text



def test_html_and_csv_exclude_o_c_by_default(tmp_path: Path):
    particle = _particle_with_elements([("O", 50.0), ("C", 25.0), ("Fe", 15.0), ("Ni", 10.0)])
    manifest = _minimal_manifest([particle])
    original_elements = copy.deepcopy(particle["energySpectrum"]["elements"])

    html_path = write_cleanliness_html(tmp_path, manifest)
    csv_path = write_cleanliness_csv(tmp_path, manifest)
    html_text = html_path.read_text(encoding="utf-8")
    assert "<th>Fe</th>" in html_text or ">Fe<" in html_text
    assert "<th>O</th>" not in html_text
    assert "<th>C</th>" not in html_text

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file))
    assert "Fe" in header
    assert "O" not in header
    assert "C" not in header

    # 报告生成不得改写原始能谱
    assert particle["energySpectrum"]["elements"] == original_elements
    assert manifest["particles"][0]["energySpectrum"]["elements"] == original_elements


def test_html_and_csv_normalize_elements_to_100_after_exclusion(tmp_path: Path):
    particle = _particle_with_elements([("O", 50.0), ("C", 25.0), ("Fe", 15.0), ("Ni", 10.0)])
    manifest = _minimal_manifest([particle])
    original_elements = copy.deepcopy(particle["energySpectrum"]["elements"])

    html_path = write_cleanliness_html(tmp_path, manifest)
    csv_path = write_cleanliness_csv(tmp_path, manifest)
    html_text = html_path.read_text(encoding="utf-8")

    # 剔除 O/C 后 Fe:Ni = 15:10 → 60/40，合计 100
    assert "<td>60.000</td>" in html_text
    assert "<td>40.000</td>" in html_text
    assert "已剔除 C、O 后归一化至 100%" in html_text

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    values = {symbol: float(rows[0][symbol]) for symbol in ("Fe", "Ni")}
    assert values == {"Fe": 60.0, "Ni": 40.0}
    assert sum(values.values()) == pytest.approx(100.0)

    # 报告生成不得改写原始能谱
    assert particle["energySpectrum"]["elements"] == original_elements
    assert manifest["particles"][0]["energySpectrum"]["elements"] == original_elements


def test_elements_normalized_to_100_even_without_exclusion(tmp_path: Path):
    particle = _particle_with_elements([("O", 50.0), ("C", 25.0), ("Fe", 15.0)])
    manifest = _minimal_manifest([particle])
    report_config = {"elementColumns": 5, "excludeElements": []}

    html_path = write_cleanliness_html(tmp_path, manifest, report_config=report_config)
    csv_path = write_cleanliness_csv(tmp_path, manifest, report_config=report_config)
    html_text = html_path.read_text(encoding="utf-8")
    assert "元素含量（wt%）已归一化至 100%；每颗粒最多显示 5 列" in html_text

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    total = sum(float(rows[0][symbol]) for symbol in ("O", "C", "Fe"))
    assert total == pytest.approx(100.0, abs=0.01)


def test_html_and_csv_include_o_c_when_exclude_empty(tmp_path: Path):
    particle = _particle_with_elements([("O", 50.0), ("C", 25.0), ("Fe", 15.0)])
    manifest = _minimal_manifest([particle])
    report_config = {"elementColumns": 5, "excludeElements": []}

    html_path = write_cleanliness_html(tmp_path, manifest, report_config=report_config)
    csv_path = write_cleanliness_csv(tmp_path, manifest, report_config=report_config)
    html_text = html_path.read_text(encoding="utf-8")
    assert "<th>O</th>" in html_text
    assert "<th>C</th>" in html_text
    assert "<th>Fe</th>" in html_text

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file))
    assert "O" in header
    assert "C" in header
    assert "Fe" in header


def test_html_includes_spectrum_image_and_hv_classification(tmp_path: Path):
    particle = _particle_with_elements([("Fe", 88.0), ("Cr", 12.0)])
    particle["energySpectrum"].update(
        {
            "spectrumImage": {
                "path": "spectra/0001.png",
                "mediaType": "image/png",
                "width": 800,
                "height": 450,
            },
            "materialClassification": {
                "materialClass": "Low Alloy Steel",
                "parentClass": "Ferrous Metals(HV<400)",
                "hardnessLabel": "HV<400",
                "hardnessBand": "Below400",
                "confidence": 1.0,
            },
        }
    )

    html_path = write_cleanliness_html(tmp_path, _minimal_manifest([particle]))
    html_text = html_path.read_text(encoding="utf-8")

    assert 'src="spectra/0001.png"' in html_text
    assert "能谱谱图" in html_text
    assert "HV 硬度分类" in html_text
    assert "HV&lt;400" in html_text
    assert "Low Alloy Steel" in html_text


def test_mock_eds_is_clearly_disclosed_in_html_and_csv(tmp_path: Path):
    particle = _particle_with_elements([("Fe", 70.0), ("Si", 30.0)])
    particle["energySpectrum"].update(
        {
            "device": "Mock EDS (acceptance test)",
            "mock": True,
            "notice": "模拟能谱，仅用于软件真机验收，不代表真实材料成分",
        }
    )
    manifest = _minimal_manifest([particle])
    manifest["acceptanceTest"] = {
        "fakeEds": True,
        "notice": "模拟能谱，仅用于软件真机验收，不代表真实材料成分",
    }

    html_path = write_cleanliness_html(tmp_path, manifest)
    csv_path = write_cleanliness_csv(tmp_path, manifest)

    html_text = html_path.read_text(encoding="utf-8")
    assert "验收模式 · 模拟能谱" in html_text
    assert "不代表真实材料成分" in html_text
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["edsDevice"] == "Mock EDS (acceptance test)"
    assert rows[0]["edsMock"] == "True"
    assert "不代表真实材料成分" in rows[0]["edsNotice"]


def test_write_cleanliness_pdf_via_edge(tmp_path: Path):
    if find_pdf_browser_executable() is None:
        pytest.skip("本机未安装 Chrome 或 Edge，跳过 HTML→PDF 测试")

    # 最小可打印 HTML（不依赖完整任务资源）
    html_path = tmp_path / "cleanliness_report.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>t</title>"
        "<style>@page{size:A4 landscape;margin:10mm}</style></head>"
        "<body><h1>清洁度检测报告 Edge 打印测试</h1><p>hello</p></body></html>",
        encoding="utf-8",
    )
    pdf_path = html_to_pdf_via_edge(html_path, tmp_path / "cleanliness_report.pdf")
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 1000


def test_build_cleanliness_reports_pdf_via_edge(tmp_path: Path):
    if find_pdf_browser_executable() is None:
        pytest.skip("本机未安装 Chrome 或 Edge，跳过清洁度报告 PDF 测试")

    spectrum_dir = tmp_path / "spectra"
    spectrum_dir.mkdir()
    (spectrum_dir / "0001.png").write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
    particle = _particle_with_elements([("Fe", 88.0), ("Cr", 12.0)])
    particle["energySpectrum"].update(
        {
            "spectrumImage": {"path": "spectra/0001.png"},
            "materialClassification": {
                "materialClass": "Low Alloy Steel",
                "hardnessLabel": "HV<400",
                "confidence": 1.0,
            },
        }
    )
    manifest = {
        "taskId": "edge-pdf-test",
        "status": "COMPLETED",
        "input": {"particleCount": 1},
        "selection": {"count": 1, "filter": {}},
        "calibration": {
            "result": {
                "equation": {"stageX": "x' = x", "stageY": "y' = y"},
                "fit": {"rmsErrorUm": 1.2},
                "validation": {"meanErrorUm": 1.5, "maxErrorUm": 2.0},
            },
            "anchors": [],
            "validation": [],
        },
        "particles": [particle],
    }
    reports = build_cleanliness_reports(tmp_path, manifest)
    assert reports["html"].is_file()
    assert reports["pdf"].is_file()
    assert reports["pdf"].stat().st_size > 1000
    html_text = reports["html"].read_text(encoding="utf-8")
    assert "size:A4 landscape" in html_text
    assert 'src="spectra/0001.png"' in html_text
    assert "HV&lt;400" in html_text
    # write_cleanliness_pdf 签名保持不变
    pdf2 = write_cleanliness_pdf(tmp_path, manifest)
    assert pdf2.stat().st_size > 1000
