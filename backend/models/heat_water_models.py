"""
MODEL for Heat and Water Shortage modules.
Both reuse the same live-data backbone (Open-Meteo) as the flood module, per the
unified-architecture design. Rule-based and transparent by default; swap in a
trained classifier later the same way flood_predictor.py does, once you have
real historical heat-mortality or water-shortage complaint data to train on.
"""
from data_sources import get_live_weather, compute_heat_index


def assess_heat_risk(lat: float, lon: float) -> dict:
    weather = get_live_weather(lat, lon)
    if not weather["ok"]:
        return {"ok": False, "error": weather["error"]}

    heat_index = compute_heat_index(weather["current_temp_c"], weather["current_humidity"])

    if heat_index >= 41:
        level = "RED"
    elif heat_index >= 35:
        level = "ORANGE"
    elif heat_index >= 30:
        level = "YELLOW"
    else:
        level = "GREEN"

    return {
        "ok": True,
        "current_temp_c": weather["current_temp_c"],
        "humidity_pct": weather["current_humidity"],
        "heat_index_c": heat_index,
        "alert_level": level,
    }


def assess_water_shortage_risk(lat: float, lon: float, recent_reservoir_pct: float = None) -> dict:
    """
    Rule-based, using real live rainfall trend (Open-Meteo) as a proxy for recharge,
    plus an optional real reservoir/groundwater % you supply from your water utility's
    real reporting (most municipal water boards publish this periodically).
    """
    weather = get_live_weather(lat, lon)
    if not weather["ok"]:
        return {"ok": False, "error": weather["error"]}

    forecast_rain_total = sum(weather["hourly_precip_mm"])  # next 24h, real forecast

    # Rainfall deficit signal: low upcoming rain = higher shortage risk pressure
    rain_component = 1.0 - min(forecast_rain_total / 20.0, 1.0)

    if recent_reservoir_pct is not None:
        reservoir_component = 1.0 - (recent_reservoir_pct / 100.0)
        score = 0.5 * rain_component + 0.5 * reservoir_component
    else:
        score = rain_component  # only rainfall signal available

    score = round(float(score), 3)

    if score >= 0.7:
        level = "RED"
    elif score >= 0.45:
        level = "ORANGE"
    elif score >= 0.25:
        level = "YELLOW"
    else:
        level = "GREEN"

    return {
        "ok": True,
        "forecast_rain_next_24h_mm": round(forecast_rain_total, 1),
        "reservoir_pct_supplied": recent_reservoir_pct,
        "shortage_risk_score": score,
        "alert_level": level,
        "note": "Supply your real reservoir/groundwater % from your municipal water "
                "board's public reporting for a more accurate score.",
    }
