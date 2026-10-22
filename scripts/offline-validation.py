from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
INFLUO_ROOT = Path(r"<path-to-optical-data>")
FORMAL_EXPORT = INFLUO_ROOT / "data" / "PhotoelectricExport.fsmpe.json"
NAVIGATION_DIR = INFLUO_ROOT / "data" / "navs"
SEM_HOLE_DIR = INFLUO_ROOT / "data" / "sem"
PARTICLE_IMAGE_DIR = INFLUO_ROOT / "captures" / "cap_roi-20250717_112100" / "images"

PARTICLE_IMAGES = (
    "0018_0031.png",
    "0018_0028.png",
    "0018_0032.png",
    "0023_0029.png",
    "0028_0002.png",
    "0023_0030.png",
    "0020_0032.png",
    "0013_0029.png",
    "0010_0008.png",
    "0006_0010.png",
)

HFW_UM = {
    "20260804151843": 1154.9,
    "20260804151926": 1154.9,
    "20260804152005": 1607.0,
    "20260804152021": 1607.0,
    "20260804152037": 1607.0,
    "20260804152107": 1607.0,
    "20260804152126": 1607.0,
    "20260804152128": 1607.0,
    "20260804152200": 1607.0,
    "20260804152228": 1295.9,
    "20260804152256": 1295.9,
    "20260804152320": 1295.9,
    "20260804152347": 1295.9,
    "20260804152419": 1607.0,
    "20260804152436": 1607.0,
    "20260804152526": 1607.0,
    "20260804152543": 2118.7,
    "20260804152626": 4703.7,
    "20260804152648": 1133.9,
    "20260804152734": 1041.0,
    "20260804152749": 1041.0,
    "20260804152933": 1041.0,
    "20260804153031": 1041.0,
}


def require_offline_data() -> None:
    required = [FORMAL_EXPORT, NAVIGATION_DIR, SEM_HOLE_DIR, PARTICLE_IMAGE_DIR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Offline validation data is missing:\n" + "\n".join(missing))


def main() -> None:
    require_offline_data()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from backend.cleanliness_core.ab_point_calibration import detect_navigation_ab
    from backend.cleanliness_core.cleanliness_detection import ParticleFilter, filter_optical_particles, yolo_seg_candidates
    from backend.cleanliness_core.hole_center_locator import HoleLocatorConfig, detect_hole
    from backend.cleanliness_core.optical_particle_analysis import load_optical_export

    output = ROOT / ".tmp" / "offline-validation"
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    export = load_optical_export(FORMAL_EXPORT)
    selected = filter_optical_particles(
        export.particles,
        ParticleFilter(min_length_um=70, max_length_um=500),
    )
    optical = {
        "particles": len(export.particles),
        "embeddedImages": sum(bool(item.image_data_base64) for item in export.particles),
        "selected70To500": len(selected),
        "coordinateSpace": export.coordinate_space,
        "registrationMode": export.registration_mode,
    }
    assert optical == {
        "particles": 389,
        "embeddedImages": 389,
        "selected70To500": 288,
        "coordinateSpace": "sourceStage",
        "registrationMode": "twoPointAB",
    }
    results["optical"] = optical
    print("formal_optical=ok particles=389 selected=288 images=389")

    fixture_model = ROOT / "resources" / "models" / "cleanliness_ab" / "fixture_det_v3_best.onnx"
    hole_model = ROOT / "resources" / "models" / "cleanliness_ab" / "hole_det_v3_best.onnx"
    navigation_rows = []
    for image_path in sorted(NAVIGATION_DIR.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        started = time.perf_counter()
        try:
            detection = detect_navigation_ab(
                image,
                fixture_model_path=fixture_model,
                hole_model_path=hole_model,
                input_size=640,
                fixture_confidence=0.1,
                hole_confidence=0.1,
                nms_iou_threshold=0.45,
                crop_margin=0.12,
                dedup_distance_px=30.0,
            )
            navigation_rows.append({
                "file": image_path.name,
                "status": "accepted",
                "fixtureConfidence": round(detection.fixture.confidence, 4),
                "holes": len(detection.holes),
                "seconds": round(time.perf_counter() - started, 3),
            })
        except ValueError as error:
            navigation_rows.append({
                "file": image_path.name,
                "status": "rejected",
                "error": str(error),
                "seconds": round(time.perf_counter() - started, 3),
            })
    accepted = [row for row in navigation_rows if row["status"] == "accepted"]
    rejected = [row for row in navigation_rows if row["status"] == "rejected"]
    assert len(navigation_rows) == 18
    assert len(accepted) == 14 and all(row["holes"] >= 2 for row in accepted)
    assert {row["file"] for row in rejected} == {
        "20260731_141619.png",
        "20260731_141628.png",
        "20260731_141636.png",
        "20260731_141642.png",
    }
    results["navigation"] = navigation_rows
    print("ab_models=ok accepted=14 rejected=4")

    particle_model = ROOT / "resources" / "models" / "cleanliness" / "particle_yolo11n_seg_reann_20260811_best.onnx"
    particle_rows = []
    for name in PARTICLE_IMAGES:
        image = cv2.imread(str(PARTICLE_IMAGE_DIR / name), cv2.IMREAD_UNCHANGED)
        started = time.perf_counter()
        candidates = yolo_seg_candidates(
            image,
            expected_length_px=150,
            expected_width_px=60,
            expected_area_px=5000,
            model_path=particle_model,
            input_size=1024,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            mask_threshold=0.5,
            maximum_detections=100,
        )
        particle_rows.append({
            "file": name,
            "detections": len(candidates),
            "seconds": round(time.perf_counter() - started, 3),
        })
    assert len(particle_rows) == 10 and all(row["detections"] > 0 for row in particle_rows)
    results["particleSegmentation"] = particle_rows
    print(f"particle_model=ok images=10 detections={sum(row['detections'] for row in particle_rows)}")

    hole_counts = {"full": 0, "arc": 0, "none": 0}
    hole_rows = []
    for stem, hfw_um in HFW_UM.items():
        image = cv2.imread(str(SEM_HOLE_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        detection = detect_hole(
            image,
            hfw_um / image.shape[1],
            config=HoleLocatorConfig(),
            info_bar_px=32,
        )
        hole_counts[detection.status] += 1
        hole_rows.append({
            "file": f"{stem}.png",
            "status": detection.status,
            "coverageDeg": round(detection.coverage_deg, 1),
            "diameterUm": None if detection.diameter_um != detection.diameter_um else round(detection.diameter_um, 1),
        })
    assert hole_counts == {"full": 17, "arc": 5, "none": 1}
    results["holeLocalization"] = hole_rows
    print("hole_localization=ok full=17 arc=5 none=1")

    (output / "offline-validation-results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"results={output / 'offline-validation-results.json'}")


if __name__ == "__main__":
    main()
