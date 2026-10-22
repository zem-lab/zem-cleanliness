from __future__ import annotations

from datetime import datetime
import hashlib
import random
from typing import Any, Mapping


class SpectrumDevice:
    """Base interface for particle energy spectrum acquisition."""

    device_name = "Unknown"
    method = "EDS"

    def acquire_particle_spectrum(
        self,
        particle: Mapping[str, Any],
        image_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class MockSpectrumDevice(SpectrumDevice):
    """Deterministic mock EDS device used before real hardware is available."""

    device_name = "Mock EDS"
    method = "EDS"

    _profiles = (
        {
            "name": "silicate",
            "elements": (("O", 42.0), ("Si", 28.0), ("Al", 12.0), ("Ca", 8.0), ("Fe", 10.0)),
        },
        {
            "name": "iron_oxide",
            "elements": (("Fe", 62.0), ("O", 27.0), ("C", 5.0), ("Si", 6.0)),
        },
        {
            "name": "carbonate",
            "elements": (("O", 48.0), ("C", 18.0), ("Ca", 26.0), ("Mg", 8.0)),
        },
        {
            "name": "alumina",
            "elements": (("O", 46.0), ("Al", 38.0), ("Si", 9.0), ("Na", 7.0)),
        },
        {
            "name": "titanium_oxide",
            "elements": (("Ti", 58.0), ("O", 32.0), ("Fe", 6.0), ("Al", 4.0)),
        },
    )

    def acquire_particle_spectrum(
        self,
        particle: Mapping[str, Any],
        image_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        seed = self._seed_for_particle(particle)
        rng = random.Random(seed)
        profile = self._profiles[seed % len(self._profiles)]

        raw_values = [
            (symbol, max(0.01, base * rng.uniform(0.82, 1.18)))
            for symbol, base in profile["elements"]
        ]
        total = sum(value for _, value in raw_values)
        elements = [
            {"symbol": symbol, "value": round(value / total * 100, 2)}
            for symbol, value in raw_values
        ]
        elements.sort(key=lambda item: item["value"], reverse=True)

        if elements:
            delta = round(100.0 - sum(item["value"] for item in elements), 2)
            elements[0]["value"] = round(elements[0]["value"] + delta, 2)

        label = str(particle.get("label", ""))
        image_path = None
        if image_context:
            image_path = image_context.get("image_path")

        return {
            "device": self.device_name,
            "method": self.method,
            "unit": "percent",
            "unit_label": "元素含量(%)",
            "particle_label": label,
            "profile": profile["name"],
            "image_path": str(image_path) if image_path else "",
            "acquired_at": datetime.now().isoformat(timespec="seconds"),
            "elements": elements,
            "mock": True,
        }

    def _seed_for_particle(self, particle: Mapping[str, Any]) -> int:
        label = str(particle.get("label", ""))
        points = particle.get("points", []) or []
        area = particle.get("area", "")
        payload = f"{label}|{len(points)}|{area}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(digest[:12], 16)


def summarize_elements(spectrum: Mapping[str, Any] | None, limit: int = 3) -> str:
    if not spectrum:
        return ""

    elements = spectrum.get("elements", [])
    if not isinstance(elements, list):
        return ""

    ordered = sorted(
        (item for item in elements if isinstance(item, Mapping)),
        key=lambda item: float(item.get("value", 0) or 0),
        reverse=True,
    )
    return ", ".join(str(item.get("symbol", "")) for item in ordered[:limit] if item.get("symbol"))
