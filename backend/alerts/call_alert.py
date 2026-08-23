"""
Real automated phone-call alerts via Twilio's Programmable Voice API.

SETUP (real steps, ~5 minutes, free trial available):
  1. Create a free account at https://www.twilio.com/try-twilio
  2. Get a free Twilio phone number from the console
  3. Copy your Account SID and Auth Token from the console dashboard
  4. Set these as environment variables before running the backend:
       export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
       export TWILIO_AUTH_TOKEN="your_auth_token"
       export TWILIO_FROM_NUMBER="+1xxxxxxxxxx"
  5. On a Twilio TRIAL account, you can only call phone numbers you've verified
     in the Twilio console (Verified Caller IDs). Upgrade the account to call
     any real number.

This module makes REAL phone calls when configured — it is not a simulation.
"""
import os
from twilio.rest import Client

ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")


def make_alert_call(to_number: str, hazard_type: str, zone_name: str, alert_level: str) -> dict:
    if not (ACCOUNT_SID and AUTH_TOKEN and FROM_NUMBER):
        return {
            "ok": False,
            "error": "Twilio not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
                     "TWILIO_FROM_NUMBER environment variables. See docstring for setup steps.",
        }

    message = (
        f"This is an automated alert from the Climate Resilient City early warning "
        f"system. A {alert_level} level {hazard_type} alert has been issued for "
        f"{zone_name}. Please take appropriate precautions. This message will now repeat. "
        f"A {alert_level} level {hazard_type} alert has been issued for {zone_name}. "
        f"Please take appropriate precautions."
    )

    twiml = f"<Response><Say voice='alice'>{message}</Say></Response>"

    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        call = client.calls.create(twiml=twiml, to=to_number, from_=FROM_NUMBER)
        return {"ok": True, "call_sid": call.sid, "status": call.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def alert_zone(zone: dict, hazard_type: str, alert_level: str) -> list:
    """Call every registered phone number for a zone. Only fires for ORANGE/RED —
    a phone call is disruptive, so we reserve it for the two highest severities.
    GREEN never reaches here (the UI disables the call action for GREEN)."""
    if alert_level not in ("ORANGE", "RED"):
        return []

    results = []
    for number in zone.get("phone_numbers", []):
        result = make_alert_call(number, hazard_type, zone["name"], alert_level)
        results.append({"number": number, **result})
    return results


def make_alert_message(to_number: str, hazard_type: str, zone_name: str, alert_level: str) -> dict:
    """Real SMS via Twilio's Programmable Messaging API. Uses the same
    TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM_NUMBER as phone calls —
    TWILIO_FROM_NUMBER must be an SMS-capable Twilio number."""
    if not (ACCOUNT_SID and AUTH_TOKEN and FROM_NUMBER):
        return {
            "ok": False,
            "error": "Twilio not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
                     "TWILIO_FROM_NUMBER environment variables. See call_alert.py docstring.",
        }

    body = (
        f"[Climate-Resilient City Alert] {alert_level} level {hazard_type} alert "
        f"issued for {zone_name}. Please take appropriate precautions and follow "
        f"instructions from local authorities."
    )

    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(body=body, to=to_number, from_=FROM_NUMBER)
        return {"ok": True, "message_sid": message.sid, "status": message.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def message_zone(zone: dict, hazard_type: str, alert_level: str, numbers: list = None) -> list:
    """Send an SMS to every registered/subscribed phone number for a zone.
    Fires for YELLOW, ORANGE, or RED — SMS is far less disruptive than a call,
    so it's available a step earlier than the phone-call escalation."""
    if alert_level not in ("YELLOW", "ORANGE", "RED"):
        return []

    target_numbers = numbers if numbers is not None else zone.get("phone_numbers", [])
    results = []
    for number in target_numbers:
        result = make_alert_message(number, hazard_type, zone["name"], alert_level)
        results.append({"number": number, **result})
    return results
