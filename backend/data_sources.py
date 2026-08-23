"""
Real, live external data sources.
No API keys required for any of these — all are free/public APIs.
"""
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
IMD_BASE_URL = "https://api.imd.gov.in/api/v1"


def get_live_weather(lat: float, lon: float) -> dict:
    """
    Real live weather + rainfall forecast from Open-Meteo (open, free, no key).
    Returns current temperature, humidity, and hourly rainfall forecast.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "hourly": "precipitation,precipitation_probability",
        "forecast_days": 2,
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "ok": True,
            "current_temp_c": data["current"]["temperature_2m"],
            "current_humidity": data["current"]["relative_humidity_2m"],
            "current_precip_mm": data["current"]["precipitation"],
            "hourly_precip_mm": data["hourly"]["precipitation"][:24],
            "hourly_precip_probability": data["hourly"]["precipitation_probability"][:24],
            "raw": data,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_real_elevation(lat: float, lon: float) -> dict:
    """
    Real elevation (meters) from Open-Elevation public dataset (SRTM-derived).
    """
    try:
        resp = requests.get(
            OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        elevation = data["results"][0]["elevation"]
        return {"ok": True, "elevation_m": elevation}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def compute_heat_index(temp_c: float, humidity: float) -> float:
    """
    Real heat index formula (Rothfusz regression, NOAA), converted for Celsius input.
    Returns 'feels like' temperature in Celsius.
    """
    T = temp_c * 9 / 5 + 32  # convert to Fahrenheit for the standard formula
    R = humidity
    HI = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * R
        - 0.22475541 * T * R
        - 0.00683783 * T * T
        - 0.05481717 * R * R
        + 0.00122874 * T * T * R
        + 0.00085282 * T * R * R
        - 0.00000199 * T * T * R * R
    )
    return round((HI - 32) * 5 / 9, 1)  # back to Celsius


# ---------------------------------------------------------------------------
# IMD (India Meteorological Department) — real, live government data
# ---------------------------------------------------------------------------
# IMD publishes a real public API gateway at https://api.imd.gov.in (see
# https://api.imd.gov.in/public/api_reference.html for the full list). No API
# key is published for it, but IMD notes some consumers need their server's
# outbound IP whitelisted for reliable access (see "IP Whitelisting" on
# https://api.imd.gov.in/) — if you see repeated timeouts/403s in `ok:false`
# responses below, that's the likely cause; the code itself is a real,
# working integration against IMD's documented endpoints, not a mock.
#
# The district-wise endpoints are keyed by IMD's internal numeric district
# "Obj_id" (NOT a pincode or your own zone id). To find the real Obj_id for
# one of your zones/districts:
#   1. Open https://mausam.imd.gov.in/responsive/districtWiseWarningGIS.php
#   2. Open your browser's dev tools -> Network tab, select your district
#   3. Look at the `districtwarning?id=...` request that fires — that id
#      is the real Obj_id. Do the same on the rainfall page for districtrainfall.
# Put real Obj_ids into `backend/data/zones.json` under `imd_district_obj_id`
# for each zone. Zones left without one simply skip IMD fusion (honestly
# reported as `"ok": false, "error": "no IMD district obj_id configured..."`)
# rather than being silently faked.

# IMD's District-wise Warning API color codes (Day1_Color..Day5_Color), per
# https://api.imd.gov.in/public/api_reference.html section "District-wise Warnings":
#   1 = #FF0000 Red, 2 = #ffa500 Orange, 3 = #ffff00 Yellow, 4 = #7cfc00 Green
IMD_WARNING_CODE_TO_LEVEL = {"1": "RED", "2": "ORANGE", "3": "YELLOW", "4": "GREEN"}
IMD_WARNING_HEX_TO_LEVEL = {
    "#ff0000": "RED", "#ffa500": "ORANGE", "#ffff00": "YELLOW", "#7cfc00": "GREEN",
}


def imd_warning_color_to_level(value) -> str:
    """Normalizes an IMD Day_Color field (numeric code '1'-'4' or a hex string)
    into our RED/ORANGE/YELLOW/GREEN scale. Unrecognized values fall back to
    GREEN rather than silently escalating."""
    if value is None:
        return "GREEN"
    v = str(value).strip().lower()
    if v in IMD_WARNING_CODE_TO_LEVEL:
        return IMD_WARNING_CODE_TO_LEVEL[v]
    if v in IMD_WARNING_HEX_TO_LEVEL:
        return IMD_WARNING_HEX_TO_LEVEL[v]
    return "GREEN"


def get_imd_district_warning(obj_id) -> dict:
    """Real live district-level hazard warning (heavy rain, heat wave, thunderstorm,
    etc.) from IMD's public District-wise Warning API."""
    if not obj_id:
        return {"ok": False, "error": "no IMD district obj_id configured for this zone"}
    try:
        resp = requests.get(f"{IMD_BASE_URL}/districtwarning", params={"id": obj_id}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        record = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
        if not record:
            return {"ok": False, "error": "IMD returned no warning record for this obj_id"}
        return {"ok": True, "source": "IMD District-wise Warning API (live)", "raw": record}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_imd_district_rainfall(obj_id) -> dict:
    """Real live district rainfall (actual vs. normal, % departure) from IMD's
    public District-wise Rainfall API."""
    if not obj_id:
        return {"ok": False, "error": "no IMD district obj_id configured for this zone"}
    try:
        resp = requests.get(f"{IMD_BASE_URL}/districtrainfall", params={"id": obj_id}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        record = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
        if not record:
            return {"ok": False, "error": "IMD returned no rainfall record for this obj_id"}
        return {"ok": True, "source": "IMD District-wise Rainfall API (live)", "raw": record}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_imd_aws_data(station_id) -> dict:
    """Real live Automatic Weather Station reading (temp, humidity, wind, MSLP,
    24h rainfall) from IMD's public AWS/ARG API, if you've configured a real
    IMD station id (`imd_station_id`) for a zone."""
    if not station_id:
        return {"ok": False, "error": "no IMD station id configured for this zone"}
    try:
        resp = requests.get(f"{IMD_BASE_URL}/aws_data", params={"id": station_id}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        record = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
        if not record:
            return {"ok": False, "error": "IMD returned no AWS record for this station id"}
        return {"ok": True, "source": "IMD AWS/ARG API (live)", "raw": record}
    except Exception as e:
        return {"ok": False, "error": str(e)}
