"""Small, verifiable country fact sheet for the flag-first application."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pycountry
import requests

from .taxonomy import country_name_from_code

REST_COUNTRIES = "https://restcountries.com/v3.1"
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
    warning: str | None = None


def _fetch_rest_country(code: str) -> dict[str, Any]:
    """Retrieve one country, accepting array and object response shapes."""
    headers = {"User-Agent": "FlagIntelligence/0.2 (country facts)"}
    for url in (
        f"{REST_COUNTRIES}/alpha/{code}?fields=name,capital,area,population,translations,flags",
        f"{REST_COUNTRIES}/alpha/{code}",
    ):
        try:
            response = requests.get(url, headers=headers, timeout=7)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                payload = payload[0] if payload else {}
            if isinstance(payload, dict) and isinstance(payload.get("name"), dict):
                return payload
        except (requests.RequestException, ValueError, IndexError, TypeError):
            continue
    return {}


def _latest_world_bank_population(code: str) -> tuple[int, str, str] | None:
    """Obtain the latest *observed* annual figure, not a simulated live counter."""
    country = pycountry.countries.get(alpha_2=code.upper())
    alpha3 = str(country.alpha_3) if country else ("XKX" if code == "xk" else "")
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


def _friendly_name(data: dict[str, Any], code: str) -> str:
    translations = data.get("translations") or {}
    french = (
        (translations.get("fra") or {}).get("common")
        if isinstance(translations, dict) else None
    )
    names = data.get("name") or {}
    common = names.get("common") if isinstance(names, dict) else None
    return str(french or common or country_name_from_code(code))


def fetch_country_facts(code: str) -> CountryFacts:
    """Best-effort country lookup, with explicit provenance/freshness."""
    code = code.strip().lower()
    if len(code) != 2 or not code.isalpha():
        raise ValueError("Country code must be two letters.")

    data = _fetch_rest_country(code)
    capital = data.get("capital")
    if isinstance(capital, list):
        capital = ", ".join(str(item) for item in capital if item)
    if not isinstance(capital, str) or not capital.strip():
        capital = None

    raw_area = data.get("area")
    area = float(raw_area) if isinstance(raw_area, (int, float)) and raw_area > 0 else None

    flag = data.get("flags") or {}
    flag_url = flag.get("png") if isinstance(flag, dict) else None
    if not isinstance(flag_url, str) or not flag_url.startswith("https://"):
        flag_url = f"https://flagcdn.com/w320/{code}.png"

    wb = _latest_world_bank_population(code)
    if wb:
        population, year, population_url = wb
        source = "Banque mondiale · SP.POP.TOTL"
        warning = None
    else:
        raw_pop = data.get("population")
        population = int(raw_pop) if isinstance(raw_pop, (int, float)) and raw_pop > 0 else None
        year = None
        source = "REST Countries (année de référence non fournie)" if population else None
        population_url = f"{REST_COUNTRIES}/alpha/{code}" if population else None
        warning = (
            "Dernière donnée annuelle de la Banque mondiale indisponible. "
            "Le chiffre de remplacement n'est pas une population en temps réel."
            if population else
            "Population indisponible auprès des sources interrogées."
        )

    if not data:
        warning = (warning + " " if warning else "") + (
            "Informations géographiques momentanément indisponibles."
        )

    return CountryFacts(
        code=code.upper(),
        name=_friendly_name(data, code),
        capital=capital,
        area_km2=area,
        population=population,
        population_year=year,
        population_source=source,
        population_source_url=population_url,
        retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        flag_url=flag_url,
        warning=warning,
    )
