"""Focused tests for the portfolio-facing flag-first flow."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from flag_recognition.flag_matcher import _prepare, rank_reference_flags
from flag_recognition import quick_facts


def _tricolor(left, center, right) -> Image.Image:
    image = Image.new("RGB", (180, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 59, 119), fill=left)
    draw.rectangle((60, 0, 119, 119), fill=center)
    draw.rectangle((120, 0, 179, 119), fill=right)
    return image


def test_distinguishes_france_from_ireland_and_ivory_coast():
    fr = _tricolor("#002395", "#ffffff", "#ed2939")
    ie = _tricolor("#169b62", "#ffffff", "#ff883e")
    ci = _tricolor("#f77f00", "#ffffff", "#009e60")
    gallery = {
        "fr": _prepare(fr),
        "ie": _prepare(ie),
        "ci": _prepare(ci),
    }
    for i in range(100):
        gallery[f"test{i}"] = np.zeros((80, 120, 3), dtype=np.uint8)
    result = rank_reference_flags(fr, gallery)
    assert result.accepted is True
    assert result.code == "fr"
    assert result.distance == 0


def test_ambiguous_similar_flags_are_not_automatically_accepted():
    identical = _tricolor("red", "white", "blue")
    gallery = {
        "a": _prepare(identical),
        "b": _prepare(identical),
    }
    for i in range(100):
        gallery[f"test{i}"] = np.zeros((80, 120, 3), dtype=np.uint8)
    result = rank_reference_flags(identical, gallery)
    assert result.accepted is False
    assert result.margin == 0


def test_population_shows_latest_available_year_and_provenance(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def raise_for_status(self):
            pass
        def json(self):
            return self.payload

    def fake_get(url, **kwargs):
        if "worldbank" in url:
            return Response([{}, [
                {"date": "2025", "value": 68_000_123},
                {"date": "2024", "value": 67_800_000},
            ]])
        return Response({
            "name": {"common": "France"},
            "translations": {"fra": {"common": "France"}},
            "capital": ["Paris"],
            "area": 551695,
            "population": 67_000_000,
            "flags": {"png": "https://flagcdn.com/w320/fr.png"},
        })

    monkeypatch.setattr(quick_facts.requests, "get", fake_get)
    facts = quick_facts.fetch_country_facts("fr")
    assert facts.name == "France"
    assert facts.capital == "Paris"
    assert facts.area_km2 == 551695.0
    assert facts.population == 68_000_123
    assert facts.population_year == "2025"
    assert "Banque mondiale" in facts.population_source


def test_population_fallback_never_claims_a_current_year(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def raise_for_status(self):
            pass
        def json(self):
            return self.payload
    def fake_get(url, **kwargs):
        if "worldbank" in url:
            return Response([{}, []])
        return Response({
            "name": {"common": "France"},
            "capital": ["Paris"],
            "population": 67_000_000,
        })
    monkeypatch.setattr(quick_facts.requests, "get", fake_get)
    facts = quick_facts.fetch_country_facts("fr")
    assert facts.population_year is None
    assert facts.population is None
    assert facts.population_source is None
    assert "Aucune valeur périmée" in facts.warning


def test_flag_first_app_is_valid_python():
    import ast
    app = Path(__file__).resolve().parents[1] / "src/flag_recognition/flag_first_app.py"
    ast.parse(app.read_text(encoding="utf-8"))
