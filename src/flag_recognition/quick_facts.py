"""Country fact sheets: bundled geography + current World Bank annual population.

The old unauthenticated REST Countries v3.1 endpoints were retired in 2026.
A local ODbL-licensed reference keeps capital and area available without keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

import pycountry
import requests

from .taxonomy import country_name_from_code

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DATA = ROOT / "data/countries_reference_min.json"
REFERENCE_URL = "https://github.com/mledoze/countries"
WORLD_BANK = "https://api.worldbank.org/v2"
WB_POPULATION_INDICATOR = "SP.POP.TOTL"


@dataclass(frozen=True)
class CountryFacts:
    code: str
    name: str
    capital: str | None
    area_km2: float | None
    population: int | None
    population_year: str | None
    population_source: str | None
    population_source_url: str | None
    retrieved_at: str
    flag_url: str | None
    geography_source: str = REFERENCE_URL
    warning: str | None = None


@lru_cache(maxsize=1)
def _reference_countries() -> dict:
    payload = json.loads(REFERENCE_DATA.read_text(encoding="utf-8"))
    countries = payload.get("countries", {})
    if not isinstance(countries, dict):
        raise ValueError("Invalid bundled country reference.")
    return countries


def _latest_world_bank_population(code: str) -> tuple[int, str, str] | None:
    """Latest observed annual statistic; never claim a live population."""
    item = pycountry.countries.get(alpha_2=code.upper())
    alpha3 = str(item.alpha_3) if item else ("XKX" if code == "xk" else "")
    if not alpha3:
        return None

    url = f"{WORLD_BANK}/country/{alpha3}/indicator/{WB_POPULATION_INDICATOR}"
    try:
        response = requests.get(
            url,
            params={"format": "json", "mrnev": 3, "per_page": 5},
            timeout=8,
            headers={"User-Agent": "FlagIntelligence/0.2 (population metadata)"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2:
            return None
        entries = payload[1]
        if not isinstance(entries, list):
            return None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            value = entry.get("value")
            year = str(entry.get("date", ""))
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
                and year.isdigit()
                and 1900 <= int(year) <= datetime.now(timezone.utc).year
            ):
                return int(round(value)), year, url + "?format=json&mrnev=3"
    except (requests.RequestException, ValueError, TypeError):
        return None
    return None


def fetch_country_facts(code: str) -> CountryFacts:
    code = code.strip().lower()
    if len(code) != 2 or not code.isalpha():
        raise ValueError("Country code must be two letters.")

    row = _reference_countries().get(code, {})
    if not isinstance(row, dict):
        row = {}
    capital_values = row.get("capital")
    if isinstance(capital_values, list):
        capital = ", ".join(str(item) for item in capital_values if item) or None
    else:
        capital = str(capital_values) if capital_values else None

    area_value = row.get("area_km2")
    area = float(area_value) if isinstance(area_value, (int, float)) and area_value > 0 else None
    name = str(row.get("name_fr") or row.get("name_en") or country_name_from_code(code))

    population_record = _latest_world_bank_population(code)
    if population_record:
        population, year, source_url = population_record
        source = "Banque mondiale · SP.POP.TOTL"
        warning = None
    else:
        population, year, source, source_url = None, None, None, None
        warning = (
            "La dernière population annuelle n'a pas pu être récupérée "
            "auprès de la Banque mondiale. Aucune valeur périmée n'est inventée."
        )

    if not row:
        warning = (warning + " " if warning else "") + (
            "La géographie de ce territoire n'est pas dans la référence locale."
        )

    return CountryFacts(
        code=code.upper(),
        name=name,
        capital=capital,
        area_km2=area,
        population=population,
        population_year=year,
        population_source=source,
        population_source_url=source_url,
        retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        flag_url=f"https://flagcdn.com/w320/{code}.png",
        geography_source=REFERENCE_URL,
        warning=warning,
    )
