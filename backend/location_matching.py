"""Country and location eligibility helpers for job recommendations."""

import re
from typing import Any

COUNTRY_CODES = {
    "india": "IN",
    "ind": "IN",
    "in": "IN",
    "united states": "US",
    "usa": "US",
    "us": "US",
    "u.s.": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "gb": "GB",
    "canada": "CA",
    "ca": "CA",
}
# The deployment does not include a country-code package (for example,
# ``pycountry``), so this is an explicit supported subset rather than a claim
# that every ISO 3166-1 alpha-2 code is accepted. Unknown values remain
# untrusted and are treated as missing location evidence.
_ISO_COUNTRY_CODES = {
    "AE",
    "AU",
    "AT",
    "BE",
    "BR",
    "CH",
    "CN",
    "DE",
    "DK",
    "ES",
    "FI",
    "FR",
    "GR",
    "HK",
    "ID",
    "IE",
    "IL",
    "IT",
    "JP",
    "KR",
    "MY",
    "MX",
    "NL",
    "NO",
    "NZ",
    "PH",
    "PL",
    "PT",
    "SE",
    "SG",
    "TH",
    "TR",
    "TW",
    "VN",
    "ZA",
    "IN",
    "US",
    "GB",
    "CA",
}
CITY_ALIASES = {"bangalore": "bengaluru", "gurgaon": "gurugram", "bombay": "mumbai"}


def _norm(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    return CITY_ALIASES.get(text, text)


def normalize_location_value(value: Any) -> str:
    """Normalize a city/state/location component without inventing data."""
    return _norm(value)


def country_code(value: Any) -> str:
    text = _norm(value)
    return COUNTRY_CODES.get(
        text, text.upper() if text.upper() in _ISO_COUNTRY_CODES else ""
    )


def _parts(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return [value.strip()] if isinstance(value, str) and value.strip() else []


def _preferred_location_parts(value: Any) -> dict[str, str] | None:
    """Parse one preferred location without broadening compound values."""
    text = str(value or "").strip()
    if not text:
        return None
    raw_parts = [part.strip() for part in text.split(",")]
    if any(not part for part in raw_parts) or len(raw_parts) > 3:
        return None

    normalized = [_norm(part) for part in raw_parts]
    if len(normalized) == 1:
        country = country_code(normalized[0])
        if country:
            return {"country_code": country}
        return {"city": normalized[0]}

    if len(normalized) == 2:
        second_country = country_code(normalized[1])
        if second_country:
            return {"state": normalized[0], "country_code": second_country}
        return {"city": normalized[0], "state": normalized[1]}

    country = country_code(normalized[2])
    if not country:
        return None
    return {
        "city": normalized[0],
        "state": normalized[1],
        "country_code": country,
    }


def candidate_location(candidate: dict[str, Any]) -> dict[str, str]:
    raw = candidate.get("raw_data") or {}
    if isinstance(raw, str):
        try:
            import json

            raw = json.loads(raw)
        except Exception:
            raw = {}
    prefs = candidate.get("_prefs_row") or {}
    preferred = prefs.get("preferred_locations") if isinstance(prefs, dict) else None
    if isinstance(preferred, str):
        try:
            import json

            preferred = json.loads(preferred)
        except Exception:
            preferred = [preferred]
    preferred = _parts(
        preferred or (raw.get("preferred_locations") if isinstance(raw, dict) else None)
    )
    explicit_country = (
        candidate.get("country_code")
        or candidate.get("country")
        or (
            raw.get("country_code") or raw.get("country")
            if isinstance(raw, dict)
            else ""
        )
    )
    explicit_city = candidate.get("city") or (
        raw.get("city") if isinstance(raw, dict) else ""
    )
    explicit_state = candidate.get("state") or (
        raw.get("state") if isinstance(raw, dict) else ""
    )
    location = candidate.get("location") or (
        raw.get("location") if isinstance(raw, dict) else ""
    )
    text = str(location or "")
    pieces = [p.strip() for p in str(text).split(",") if p.strip()]
    if not explicit_country and pieces:
        explicit_country = pieces[-1]
    if not explicit_city and pieces:
        explicit_city = pieces[0]
    if not explicit_state and len(pieces) >= 3:
        explicit_state = pieces[-2]
    return {
        "country_code": country_code(explicit_country),
        "city": _norm(explicit_city),
        "state": _norm(explicit_state),
        "preferred": " | ".join(preferred),
    }


def job_location(job: dict[str, Any]) -> dict[str, Any]:
    structured = job.get("structured_data")
    if not isinstance(structured, dict):
        structured = {}
    raw_source = structured.get("raw_source")
    payload = raw_source.get("payload") if isinstance(raw_source, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    location_details = structured.get("location_details")
    if not isinstance(location_details, dict):
        location_details = {}
    raw = (
        job.get("location")
        or location_details.get("location")
        or payload.get("location")
        or ""
    )
    pieces = [p.strip() for p in str(raw).split(",")]
    inferred_country = ""
    inferred_city = ""
    inferred_state = ""
    if pieces and all(pieces) and len(pieces) == 3:
        inferred_city, inferred_state, inferred_country = pieces[-3:]
    elif pieces and all(pieces) and len(pieces) == 2:
        if country_code(pieces[1]):
            inferred_state, inferred_country = pieces
        else:
            inferred_city, inferred_state = pieces
    elif pieces and all(pieces) and len(pieces) == 1:
        inferred_city = pieces[0]
    country = (
        job.get("country")
        or location_details.get("country")
        or payload.get("country")
        or inferred_country
    )
    city = (
        job.get("city")
        or location_details.get("city")
        or payload.get("city")
        or inferred_city
    )
    state = (
        job.get("state")
        or location_details.get("state")
        or payload.get("state")
        or inferred_state
    )
    policy = _norm(job.get("remote_policy"))
    scope = country_code(country)
    if "worldwide" in policy or "global" in policy:
        scope = "WORLDWIDE"
    locations = (
        job.get("locations")
        or location_details.get("locations")
        or payload.get("locations")
    )
    if not isinstance(locations, (list, tuple, set)):
        locations = [raw] if raw else []
    parsed_locations = []
    for value in locations:
        value = str(value or "").strip()
        if not value:
            continue
        parts = [part.strip() for part in value.split(",")]
        if len(parts) == 3 and all(parts):
            parsed_locations.append(
                {
                    "city": _norm(parts[0]),
                    "state": _norm(parts[1]),
                    "country_code": country_code(parts[2]),
                }
            )
        elif len(parts) == 2 and all(parts):
            if country_code(parts[1]):
                parsed_locations.append(
                    {"state": _norm(parts[0]), "country_code": country_code(parts[1])}
                )
            else:
                parsed_locations.append(
                    {"city": _norm(parts[0]), "state": _norm(parts[1])}
                )
        elif len(parts) == 1 and parts[0]:
            parsed_locations.append({"city": _norm(parts[0])})
    primary = {"city": _norm(city), "state": _norm(state), "country_code": scope}
    if primary not in parsed_locations and any(primary.values()):
        parsed_locations.insert(0, primary)
    return {
        "country_code": scope,
        "city": _norm(city),
        "state": _norm(state),
        "remote": "true"
        if job.get("remote") is True or "remote" in policy
        else "false",
        "policy": policy,
        "locations": parsed_locations,
    }


def country_eligible(candidate: dict[str, Any], job: dict[str, Any]) -> bool:
    c, j = candidate_location(candidate), job_location(job)
    job_countries = {
        item.get("country_code")
        for item in j.get("locations", [])
        if item.get("country_code")
    }
    if j["country_code"]:
        job_countries.add(j["country_code"])
    if not c["country_code"] or not job_countries:
        return True
    if "WORLDWIDE" in job_countries:
        return True
    return c["country_code"] in job_countries


def location_preference_matches(preferences: Any, job: dict[str, Any]) -> bool:
    """Match preferred locations by structured city/state/country values."""
    j = job_location(job)
    candidates = j.get("locations") or [j]
    if not any(
        item.get("city") or item.get("state") or item.get("country_code")
        for item in candidates
    ):
        return False
    for preferred in _parts(preferences):
        parsed = _preferred_location_parts(preferred)
        if parsed is not None:
            for actual in candidates:
                if all(actual.get(key, "") == value for key, value in parsed.items()):
                    return True
    return False


def location_preference_score(candidate: dict[str, Any], job: dict[str, Any]) -> float:
    c, j = candidate_location(candidate), job_location(job)
    if not country_eligible(candidate, job):
        return 0.0
    locations = j.get("locations") or [j]
    if any(c["city"] and c["city"] == item.get("city") for item in locations):
        return 1.0
    if any(c["state"] and c["state"] == item.get("state") for item in locations):
        return 0.8
    return 0.5
