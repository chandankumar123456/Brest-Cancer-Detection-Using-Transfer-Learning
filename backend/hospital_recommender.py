"""Hospital recommendation service with optional LLM summarization."""

from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _http_get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 15) -> Any:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 20) -> Any:
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=merged_headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geocode_location(location_query: str) -> tuple[float, float, str]:
    """Resolve free-text location to latitude/longitude using Nominatim."""
    params = urllib.parse.urlencode({"q": location_query, "format": "json", "limit": 1})
    url = f"{NOMINATIM_URL}?{params}"
    headers = {"User-Agent": "OnchoScanAI/1.0 (breast-cancer-research)"}
    results = _http_get_json(url, headers=headers, timeout=15)

    if not isinstance(results, list) or len(results) == 0:
        raise ValueError("Location not found. Try a more specific city or area.")

    first = results[0]
    lat = float(first["lat"])
    lon = float(first["lon"])
    label = str(first.get("display_name", location_query))
    return lat, lon, label


def fetch_nearby_hospitals(lat: float, lon: float, radius_km: int = 35, limit: int = 8) -> list[dict[str, Any]]:
    """Query nearby hospitals from OpenStreetMap Overpass API."""
    radius_m = int(radius_km * 1000)
    query = f"""
[out:json][timeout:25];
(
  node["amenity"="hospital"](around:{radius_m},{lat},{lon});
  way["amenity"="hospital"](around:{radius_m},{lat},{lon});
  relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
);
out center tags;
""".strip()

    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(OVERPASS_URL, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)

    elements = data.get("elements", []) if isinstance(data, dict) else []
    hospitals: list[dict[str, Any]] = []
    for element in elements:
        tags = element.get("tags", {})
        if not tags:
            continue

        e_lat = element.get("lat") or element.get("center", {}).get("lat")
        e_lon = element.get("lon") or element.get("center", {}).get("lon")
        if e_lat is None or e_lon is None:
            continue

        distance = _haversine_km(lat, lon, float(e_lat), float(e_lon))
        hospitals.append(
            {
                "name": tags.get("name", "Unnamed Hospital"),
                "distance_km": round(distance, 2),
                "address": ", ".join(
                    p
                    for p in [
                        tags.get("addr:housenumber"),
                        tags.get("addr:street"),
                        tags.get("addr:city"),
                        tags.get("addr:state"),
                    ]
                    if p
                )
                or "Address not available",
                "phone": tags.get("phone") or tags.get("contact:phone") or "N/A",
                "website": tags.get("website") or tags.get("contact:website") or "N/A",
                "emergency": tags.get("emergency", "unknown"),
            }
        )

    hospitals.sort(key=lambda item: item["distance_km"])
    return hospitals[:limit]


def _format_hospitals_for_prompt(hospitals: list[dict[str, Any]]) -> str:
    lines = []
    for i, item in enumerate(hospitals, start=1):
        lines.append(
            (
                f"{i}. {item['name']} | distance: {item['distance_km']} km | "
                f"address: {item['address']} | phone: {item['phone']} | website: {item['website']}"
            )
        )
    return "\n".join(lines)


def summarize_with_ollama(prompt: str) -> str:
    """Use local Ollama model when available."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful medical navigation assistant. Do not diagnose. Recommend next-care logistics only.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    response = _http_post_json("http://127.0.0.1:11434/api/chat", payload, timeout=25)
    return response["message"]["content"].strip()


def summarize_with_openai_compatible(prompt: str) -> str:
    """Use any OpenAI-compatible endpoint through env vars."""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a hospital recommendation assistant for breast cancer follow-up. Be concise, practical, and safety-focused.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    response = _http_post_json(f"{base_url}/chat/completions", payload, headers=headers, timeout=25)
    return response["choices"][0]["message"]["content"].strip()


def build_recommendation_prompt(location_label: str, diagnosis: str, hospitals: list[dict[str, Any]]) -> str:
    return (
        "Patient context:\n"
        f"- location: {location_label}\n"
        f"- model output: {diagnosis}\n\n"
        "Nearby hospitals:\n"
        f"{_format_hospitals_for_prompt(hospitals)}\n\n"
        "Task:\n"
        "1) Provide a short next-step recommendation focused on logistics and specialist follow-up.\n"
        "2) Mention 2-3 best options from the list with reasons (distance, contact completeness).\n"
        "3) Add a warning that this is not medical diagnosis and emergency symptoms need urgent care.\n"
        "Keep under 140 words."
    )


def fallback_summary(location_label: str, diagnosis: str, hospitals: list[dict[str, Any]]) -> str:
    top_names = ", ".join(item["name"] for item in hospitals[:3])
    return (
        f"Based on your location ({location_label}) and AI result ({diagnosis}), prioritize contacting nearby hospitals first: "
        f"{top_names}. Ask for oncology or breast clinic availability, pathology review support, and earliest appointment slots. "
        "This recommendation is logistical support only and is not a medical diagnosis. If symptoms are severe or rapidly worsening, seek emergency care immediately."
    )


def get_hospital_recommendations(location_query: str, diagnosis: str, radius_km: int = 35) -> dict[str, Any]:
    """Fetch location-based hospitals and add LLM recommendation summary."""
    lat, lon, location_label = geocode_location(location_query)
    hospitals = fetch_nearby_hospitals(lat, lon, radius_km=radius_km, limit=8)
    if not hospitals:
        raise ValueError("No hospitals found near this location. Try increasing search radius.")

    prompt = build_recommendation_prompt(location_label, diagnosis, hospitals)
    llm_used = "none"

    summary_text = ""
    try:
        summary_text = summarize_with_openai_compatible(prompt)
        llm_used = os.getenv("LLM_MODEL", "gpt-4o-mini")
    except Exception as openai_exc:
        logger.info("OpenAI-compatible LLM unavailable: %s", openai_exc)
        try:
            summary_text = summarize_with_ollama(prompt)
            llm_used = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        except Exception as ollama_exc:
            logger.info("Ollama LLM unavailable: %s", ollama_exc)
            summary_text = fallback_summary(location_label, diagnosis, hospitals)

    return {
        "location": {
            "query": location_query,
            "resolved_name": location_label,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
        },
        "diagnosis": diagnosis,
        "llm_model_used": llm_used,
        "summary": summary_text,
        "hospitals": hospitals,
    }
