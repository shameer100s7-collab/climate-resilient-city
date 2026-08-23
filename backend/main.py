import asyncio
import json
import os
import shutil
import subprocess
import sys

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))

from data_sources import (
    get_live_weather, get_real_elevation,
    get_imd_district_warning, get_imd_district_rainfall, get_imd_aws_data,
    imd_warning_color_to_level,
)
from models.flood_predictor import predict_flood_risk
from models.heat_water_models import assess_heat_risk, assess_water_shortage_risk
from models.cv_flood_detector import analyze_frame
from alerts.call_alert import alert_zone, message_zone
import subscribers

app = FastAPI(title="Climate-Resilient City Early Warning System")

# RED > ORANGE > YELLOW > GREEN — used to fuse the internal model's alert
# level with IMD's own issued warning level (whichever is more severe wins,
# so a live government warning can never be silently overridden downward).
LEVEL_SEVERITY = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}


def more_severe(level_a: str, level_b: str) -> str:
    return level_a if LEVEL_SEVERITY.get(level_a, 0) >= LEVEL_SEVERITY.get(level_b, 0) else level_b

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ZONES_PATH = os.path.join(os.path.dirname(__file__), "data", "zones.json")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

GIS_ADDON_DIR = os.path.join(os.path.dirname(__file__), "gis_addon")
GIS_ADDON_DATA_DIR = os.path.join(GIS_ADDON_DIR, "data")
GIS_ADDON_GRID_CSV = os.path.join(GIS_ADDON_DATA_DIR, "composite_flood_risk_grid.csv")
GIS_ADDON_HEATMAP = os.path.join(GIS_ADDON_DATA_DIR, "flood_risk_heatmap.png")

# In-memory store for latest IoT sensor readings and last CV detections, keyed by zone id.
IOT_READINGS = {}
CV_DETECTIONS = {}

# Cache real elevation per zone (it never changes, no need to re-fetch every refresh)
ELEVATION_CACHE = {}


def get_cached_elevation(zone: dict) -> float:
    if "elevation_m" in zone:
        return zone["elevation_m"]  # user supplied a real known value in zones.json
    if zone["id"] in ELEVATION_CACHE:
        return ELEVATION_CACHE[zone["id"]]
    result = get_real_elevation(zone["lat"], zone["lon"])
    elevation = result["elevation_m"] if result["ok"] else 10  # last-resort fallback only if API fails
    ELEVATION_CACHE[zone["id"]] = elevation
    return elevation


def load_zones():
    with open(ZONES_PATH) as f:
        return json.load(f)["zones"]


def compute_zone(zone: dict) -> dict:
    """Computes the full fused hazard payload for one zone. Shared by /api/zones
    and the background auto-notify loop so both see identical, consistent data."""
    weather = get_live_weather(zone["lat"], zone["lon"])
    rainfall_next_6h = sum(weather["hourly_precip_mm"][:6]) if weather["ok"] else 0

    flood = predict_flood_risk(
        rainfall_mm_next_6h=rainfall_next_6h,
        elevation_m=get_cached_elevation(zone),
        drainage_capacity_score=zone["drainage_capacity_score"],
        historical_flood_frequency=zone["historical_flood_frequency"],
    )

    # If an IoT sensor or CCTV has reported real data for this zone, it overrides/boosts confidence
    iot = IOT_READINGS.get(zone["id"])
    cv = CV_DETECTIONS.get(zone["id"])
    confidence = "medium_prediction_only"
    if cv:
        confidence = "high_cv_confirmed"
        if cv.get("severity") in ("MODERATE", "SEVERE"):
            flood["alert_level"] = "RED"
    elif iot:
        confidence = "high_sensor_confirmed"

    # Live IMD district warning, fused in: whichever of our model vs. IMD's own
    # issued warning is more severe wins the flood zone's displayed alert_level.
    # Honest no-op if the zone has no imd_district_obj_id configured yet.
    imd_warning = get_imd_district_warning(zone.get("imd_district_obj_id"))
    if imd_warning["ok"]:
        imd_level = imd_warning_color_to_level(imd_warning["raw"].get("Day1_Color"))
        imd_warning["day1_level"] = imd_level
        imd_warning["day1_warning_text"] = imd_warning["raw"].get("Day_1")
        fused_level = more_severe(flood["alert_level"], imd_level)
        if fused_level != flood["alert_level"]:
            flood["alert_level"] = fused_level
            flood["alert_level_source"] = "IMD district warning (live, more severe than internal model)"

    heat = assess_heat_risk(zone["lat"], zone["lon"])
    water = assess_water_shortage_risk(zone["lat"], zone["lon"])

    return {
        "id": zone["id"],
        "name": zone["name"],
        "lat": zone["lat"],
        "lon": zone["lon"],
        "flood": {**flood, "confidence": confidence},
        "heat": heat,
        "water": water,
        "iot_reading": iot,
        "cv_detection": cv,
        "imd_warning": imd_warning,
        "subscriber_count": len(subscribers.get_subscribers(zone["id"])),
    }


@app.get("/api/zones")
def get_zones():
    """Full dashboard payload: every zone with live flood/heat/water risk fused
    in, including IMD's own live district warnings where configured."""
    zones = load_zones()
    return {"zones": [compute_zone(z) for z in zones]}


@app.get("/api/imd/district-warning")
def imd_district_warning(zone_id: str):
    zones = load_zones()
    zone = next((z for z in zones if z["id"] == zone_id), None)
    if not zone:
        return {"ok": False, "error": "zone not found"}
    return get_imd_district_warning(zone.get("imd_district_obj_id"))


@app.get("/api/imd/district-rainfall")
def imd_district_rainfall(zone_id: str):
    zones = load_zones()
    zone = next((z for z in zones if z["id"] == zone_id), None)
    if not zone:
        return {"ok": False, "error": "zone not found"}
    return get_imd_district_rainfall(zone.get("imd_district_obj_id"))


@app.get("/api/imd/aws")
def imd_aws(zone_id: str):
    zones = load_zones()
    zone = next((z for z in zones if z["id"] == zone_id), None)
    if not zone:
        return {"ok": False, "error": "zone not found"}
    return get_imd_aws_data(zone.get("imd_station_id"))


@app.get("/api/elevation")
def elevation(lat: float, lon: float):
    return get_real_elevation(lat, lon)


@app.post("/api/iot/ingest")
def iot_ingest(zone_id: str, water_level_cm: float, sensor_id: str = "unknown"):
    """Real endpoint for physical IoT sensors (ultrasonic/float sensors at drain
    choke points) to POST readings to. Point your sensor's HTTP client here."""
    IOT_READINGS[zone_id] = {
        "water_level_cm": water_level_cm,
        "sensor_id": sensor_id,
    }
    return {"ok": True, "stored": IOT_READINGS[zone_id]}


@app.post("/api/cv/analyze")
async def cv_analyze(zone_id: str, file: UploadFile = File(...)):
    """Upload a CCTV frame (jpg/png) for this zone; runs the real CV water detector."""
    save_path = os.path.join(UPLOAD_DIR, f"{zone_id}_{file.filename}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_frame(save_path)
    if result.get("ok"):
        CV_DETECTIONS[zone_id] = result
    return result


def _zone_and_level(zone_id: str, hazard_type: str):
    zones = load_zones()
    zone = next((z for z in zones if z["id"] == zone_id), None)
    if not zone:
        return None, None
    # Reuse the fused computation (internal model + live IMD warning) so the
    # level used to fire a call/SMS always matches what officials see on screen.
    computed = compute_zone(zone)
    level = computed[hazard_type].get("alert_level", "GREEN")
    return zone, level


@app.post("/api/alerts/trigger")
def trigger_alerts(zone_id: str, hazard_type: str):
    """Fires real phone calls (via Twilio) to every number registered for this
    zone. The UI disables this action entirely at GREEN; the backend also
    enforces it (alert_zone only fires for ORANGE/RED) as a second line of defense."""
    zone, level = _zone_and_level(zone_id, hazard_type)
    if not zone:
        return {"ok": False, "error": "zone not found"}

    call_results = alert_zone(zone, hazard_type, level)
    return {"ok": True, "alert_level": level, "calls": call_results}


@app.post("/api/alerts/message")
def trigger_message(zone_id: str, hazard_type: str):
    """Fires a real SMS (via Twilio) to every number registered for this zone,
    for YELLOW/ORANGE/RED. This is the lighter-weight action available a step
    before a phone call is warranted."""
    zone, level = _zone_and_level(zone_id, hazard_type)
    if not zone:
        return {"ok": False, "error": "zone not found"}

    message_results = message_zone(zone, hazard_type, level)
    return {"ok": True, "alert_level": level, "messages": message_results}


class SubscribeRequest(BaseModel):
    zone_id: str
    phone_number: str


@app.post("/api/subscribe")
def subscribe(req: SubscribeRequest):
    """Citizen opt-in: registers a real phone number to receive an automatic
    SMS whenever this zone's alert level rises to YELLOW/ORANGE/RED (see the
    background auto-notify loop below)."""
    zones = load_zones()
    if not any(z["id"] == req.zone_id for z in zones):
        return {"ok": False, "error": "zone not found"}
    return subscribers.subscribe(req.zone_id, req.phone_number)


@app.post("/api/unsubscribe")
def unsubscribe(req: SubscribeRequest):
    return subscribers.unsubscribe(req.zone_id, req.phone_number)


@app.get("/api/subscribers")
def subscriber_counts():
    """For the government-official UI: how many citizens are subscribed per zone."""
    return {"ok": True, "counts": subscribers.get_all_counts()}


# --- GIS Add-on Module (DEM flow accumulation + encroachment + drainage deficit + rainfall fusion) ---
# This is a separate, richer feature-engineering pipeline for Model 1, contributed as a standalone
# module. It runs on synthetic data until you follow gis_addon/README.md to swap in real DEM/
# rainfall/water-mask files. It does NOT auto-fuse into /api/zones because its output grid uses
# row/col indices that only correspond to real lat/lon once you load a real geo-referenced DEM
# (see the README's note on replacing the placeholder station-to-grid mapping with a proper
# lat/lon spatial join). Until then, treat it as a separate advanced-analysis view.

@app.get("/api/gis-addon/status")
def gis_addon_status():
    if not os.path.exists(GIS_ADDON_GRID_CSV):
        return {"ok": True, "has_run": False,
                "message": "Not run yet. POST to /api/gis-addon/run to execute it."}

    import pandas as pd
    df = pd.read_csv(GIS_ADDON_GRID_CSV)
    tiers = df["risk_tier"].value_counts().to_dict()
    top_cells = df.sort_values("composite_risk", ascending=False).head(10).to_dict(orient="records")
    return {
        "ok": True,
        "has_run": True,
        "data_source_warning": "Reflects whatever data the pipeline scripts are currently "
                                "configured with. Synthetic by default -- see "
                                "gis_addon/README.md to plug in real DEM/rainfall data.",
        "risk_tier_distribution": tiers,
        "top_risk_cells": top_cells,
    }


@app.post("/api/gis-addon/run")
def gis_addon_run():
    """Executes the GIS add-on pipeline (rainfall categorization -> flow
    accumulation -> encroachment/drainage vulnerability -> fusion -> heatmap)."""
    os.makedirs(GIS_ADDON_DATA_DIR, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "run_all.py"],
        cwd=GIS_ADDON_DIR, capture_output=True, text=True, timeout=300,
    )
    return {
        "ok": result.returncode == 0,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:] if result.returncode != 0 else None,
    }


@app.get("/api/gis-addon/heatmap")
def gis_addon_heatmap():
    if not os.path.exists(GIS_ADDON_HEATMAP):
        return {"ok": False, "error": "Heatmap not generated yet. POST to /api/gis-addon/run first."}
    return FileResponse(GIS_ADDON_HEATMAP)


# --- Background auto-notify loop -------------------------------------------------
# Polls live data (weather + IMD) for every zone/hazard on an interval and fires a
# real SMS to that zone's citizen subscribers automatically the moment a hazard
# *newly* rises to YELLOW/ORANGE/RED (edge-triggered off LAST_NOTIFIED_LEVEL, so
# subscribers aren't re-texted every cycle while a level holds steady). This is
# what makes "get real-time data automatically" actually automatic, rather than
# only updating when someone has the dashboard open.

AUTO_NOTIFY_INTERVAL_SECONDS = int(os.environ.get("AUTO_NOTIFY_INTERVAL_SECONDS", "300"))
LAST_NOTIFIED_LEVEL = {}  # keyed by f"{zone_id}:{hazard_type}" -> last alert_level sent


async def auto_notify_loop():
    while True:
        try:
            zones = load_zones()
            for zone in zones:
                computed = compute_zone(zone)
                for hazard_type in ("flood", "heat", "water"):
                    level = computed[hazard_type].get("alert_level", "GREEN")
                    key = f"{zone['id']}:{hazard_type}"
                    previous = LAST_NOTIFIED_LEVEL.get(key, "GREEN")
                    if level in ("YELLOW", "ORANGE", "RED") and level != previous:
                        subs = subscribers.get_subscribers(zone["id"])
                        if subs:
                            message_zone(zone, hazard_type, level, numbers=subs)
                        LAST_NOTIFIED_LEVEL[key] = level
                    elif level == "GREEN":
                        LAST_NOTIFIED_LEVEL[key] = "GREEN"
        except Exception as e:
            print(f"[auto_notify_loop] error: {e}")
        await asyncio.sleep(AUTO_NOTIFY_INTERVAL_SECONDS)


@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(auto_notify_loop())


# Serve the frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
