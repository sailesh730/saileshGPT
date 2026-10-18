import math
import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from database import SessionLocal, save_memory, get_memories
load_dotenv()
from rag import retrive_from_rag
from urllib.parse import urlparse
from ipaddress import ip_address

from playwright.sync_api import sync_playwright

# like a chatpt 


CUREENT_THRED_ID = "defults"

def thred_id_set(thread_id:str):
    global CUREENT_THRED_ID
    CUREENT_THRED_ID = thread_id



wed_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic"
)


@tool 
def calculator(expression:str)-> str:
    """usedful fro simple mah calcualtion 
    input should be a valid math expression
    expample 10+10 = 20 mean of 4354 math.sqrt(160) 1314131*311"""

    try:
        allowed={
            "math":math,
            "abs":abs,
            "round":round,
            "min":min,
            "max":max,
            "sun":sum
        }
        result = eval(expression,{"__builtins__":{}},allowed)
        return str(result)
    except Exception as e:
        return f"caclutation error :{str(e)}"


@tool
def rember_this(memory: str) -> str:
    """Save an important user fact or preference into long-term memory."""
    db = SessionLocal()
    try:
        item = save_memory(db, user_id=CUREENT_THRED_ID, memory=memory, memory_type="note")
        return f"Saved memory: {item.memory}"
    finally:
        db.close()


@tool
def recall_memory(query: str) -> str:
    """Recall saved long-term memories relevant to the current conversation."""
    db = SessionLocal()
    try:
        memories = get_memories(db, user_id=CUREENT_THRED_ID)
        if not memories:
            return "No saved memories found for this session."

        matching = []
        q = (query or "").lower()
        for item in memories:
            text = item.memory.lower()
            if q in text or any(word in text for word in q.split() if word):
                matching.append(item.memory)

        if not matching:
            return "No saved memories match that query."
        return "\n\n".join(f"- {m}" for m in matching)
    finally:
        db.close()


# weather_tool.py








OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

URL = "https://api.openweathermap.org/data/2.5/weather"


@tool
def get_weather(city: str) -> dict:
    """
    Get the current weather for a city.

    Returns:
    - Temperature
    - Feels-like temperature
    - Humidity
    - Atmospheric pressure
    - Visibility
    - Wind speed
    - Weather description
    - City and country

    Use this tool when the user asks about current weather,
    temperature, humidity, pressure, visibility, wind,
    or weather conditions.
    """

    if not OPENWEATHER_API_KEY:
        return {
            "success": False,
            "error": "OPENWEATHER_API_KEY is not configured."
        }

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(
            URL,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        return {
            "success": True,
            "city": data.get("name"),
            "country": data.get("sys", {}).get("country"),

            "temperature_c": data.get("main", {}).get("temp"),
            "feels_like_c": data.get("main", {}).get("feels_like"),

            "humidity_percent": data.get("main", {}).get("humidity"),
            "pressure_hpa": data.get("main", {}).get("pressure"),

            "visibility_meters": data.get("visibility"),

            "wind_speed_mps": data.get("wind", {}).get("speed"),

            "weather": data.get("weather", [{}])[0].get("description"),

        }

    except requests.exceptions.HTTPError as e:

        return {
            "success": False,
            "error": f"OpenWeather API error: {e}"
        }

    except requests.exceptions.RequestException as e:

        return {
            "success": False,
            "error": f"Network error: {e}"
        }

    except Exception as e:

        return {
            "success": False,
            "error": f"Unexpected error: {e}"
        }
@tool
def search_uploaded_document(query:str, config:RunnableConfig) -> str:
    """
    search oploaded documenet for relevent information 
    used this when the user ske about opoaed pdf,dock ,txt ,notes,files , or documenet"""
    thread_id = config.get("configurable", {}).get("thread_id", CUREENT_THRED_ID)
    return retrive_from_rag(
        query= query,
        thread_id = thread_id
    )






# for brower automation
def validate_url(url: str) -> bool:
    """Allow public HTTP/HTTPS website URLs only."""
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            print("This is an not https or http URL")
            return False

        if not parsed.hostname or parsed.username or parsed.password:
            print("plz login this wedsite before using so, i can acess")
            return False

        host = parsed.hostname.lower()

        if host == "localhost" or host.endswith(".localhost"):
            return False

        # Reject direct connections to private/reserved IPs.
        try:
            ip = ip_address(host)
            if not ip.is_global:
                return False
        except ValueError:
            pass  # Hostname; DNS must also be checked in production.

        return True

    except (ValueError, TypeError):
        return False


@tool
def browser_read(url: str) -> str:
    """Open a public website and return its title and readable text.
    Use this when the user asks the AI to visit or read a webpage.
    Input: a complete HTTP or HTTPS URL.
    """
    if not validate_url(url):
        return "Invalid URL. Provide a public HTTP/HTTPS website."

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            page = browser.new_page()
            page.set_default_timeout(15000)

            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=20000,
            )

            if response is None:
                result = "The website did not return a response."
            elif response.status >= 400:
                result = f"Website returned HTTP {response.status}."
            else:
                title = page.title()
                text = page.locator("body").inner_text(
                    timeout=10000
                )

                result = (
                    f"Website: {page.url}\n"
                    f"Title: {title}\n\n"
                    f"Content:\n{text[:10000]}"
                )

            browser.close()
            return result

    except Exception as e:
        return f"Browser error: {type(e).__name__}: {e}"



#for travel agent 
"""
travel_agent.py - an AI travel assistant that can answer things like:

  "Give me a 7 day trip plan from Nepal to UAE"
  "What are the main airports in Thailand?"
  "Status of flight EK203"
  "What's the weather in Dubai next week and how much is 500 USD in AED?"

Setup:
    pip install requests python-dotenv langchain langchain-core langgraph \
                langchain-anthropic

.env file:
    AVIATIONSTACK_API_KEY=your_aviationstack_key
    ANTHROPIC_API_KEY=your_anthropic_key

Web search is NOT included here. Plug in your own search tool(s) through
EXTRA_TOOLS near the bottom of this file.

Free APIs used without a key: REST Countries, Open-Meteo, open.er-api.com.
"""

import os
from datetime import datetime
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

AVIATIONSTACK_KEY = os.getenv("AVIATIONSTACK_API_KEY")
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1"  # free plan = http only
RESTCOUNTRIES = "https://restcountries.com/v3.1"
OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
FX_URL = "https://open.er-api.com/v6/latest"
TIMEOUT = 15
VALID_STATUSES = {"scheduled", "active", "landed", "cancelled", "incident", "diverted"}


# ==========================================================================
# Static data
# ==========================================================================
COUNTRY_ALIASES = {
    "uae": "AE", "emirates": "AE", "uk": "GB", "britain": "GB", "england": "GB",
    "usa": "US", "america": "US", "korea": "KR", "south korea": "KR",
    "russia": "RU", "vietnam": "VN", "holland": "NL",
}

# Regional blocs (by ISO alpha-2 code). Static list, so double-check for official use.
BLOCS = {
    "GCC (Gulf Cooperation Council)": {"BH", "KW", "OM", "QA", "SA", "AE"},
    "SAARC": {"AF", "BD", "BT", "IN", "MV", "NP", "PK", "LK"},
    "BIMSTEC": {"BD", "BT", "IN", "MM", "NP", "LK", "TH"},
    "ASEAN": {"BN", "KH", "ID", "LA", "MY", "MM", "PH", "SG", "TH", "VN", "TL"},
}

# (IATA, name, city, note)
MAIN_AIRPORTS = {
    "NP": [
        ("KTM", "Tribhuvan International Airport", "Kathmandu", "Main international gateway"),
        ("PKR", "Pokhara International Airport", "Pokhara", "International and domestic"),
        ("BWA", "Gautam Buddha International Airport", "Bhairahawa (Lumbini)", "International and domestic"),
        ("BIR", "Biratnagar Airport", "Biratnagar", "Domestic, eastern Nepal"),
    ],
    "AE": [
        ("DXB", "Dubai International Airport", "Dubai", "World's busiest for international passengers"),
        ("AUH", "Zayed International Airport", "Abu Dhabi", "Etihad hub"),
        ("SHJ", "Sharjah International Airport", "Sharjah", "Air Arabia hub, budget carriers"),
        ("DWC", "Al Maktoum International Airport", "Dubai South", "Cargo and some passenger flights"),
        ("RKT", "Ras Al Khaimah International Airport", "Ras Al Khaimah", "Regional and budget flights"),
        ("FJR", "Fujairah International Airport", "Fujairah", "East coast"),
    ],
    "IN": [
        ("DEL", "Indira Gandhi International Airport", "Delhi", "Largest hub"),
        ("BOM", "Chhatrapati Shivaji Maharaj International Airport", "Mumbai", ""),
        ("BLR", "Kempegowda International Airport", "Bengaluru", ""),
        ("MAA", "Chennai International Airport", "Chennai", ""),
        ("CCU", "Netaji Subhas Chandra Bose International Airport", "Kolkata", ""),
        ("HYD", "Rajiv Gandhi International Airport", "Hyderabad", ""),
    ],
    "US": [
        ("JFK", "John F. Kennedy International Airport", "New York", ""),
        ("LAX", "Los Angeles International Airport", "Los Angeles", ""),
        ("ORD", "O'Hare International Airport", "Chicago", ""),
        ("ATL", "Hartsfield-Jackson Atlanta International Airport", "Atlanta", ""),
        ("SFO", "San Francisco International Airport", "San Francisco", ""),
        ("DFW", "Dallas/Fort Worth International Airport", "Dallas", ""),
    ],
    "GB": [
        ("LHR", "Heathrow Airport", "London", "Main hub"),
        ("LGW", "Gatwick Airport", "London", ""),
        ("STN", "Stansted Airport", "London", "Budget carriers"),
        ("MAN", "Manchester Airport", "Manchester", ""),
        ("EDI", "Edinburgh Airport", "Edinburgh", ""),
    ],
    "QA": [("DOH", "Hamad International Airport", "Doha", "Qatar Airways hub")],
    "SA": [
        ("RUH", "King Khalid International Airport", "Riyadh", ""),
        ("JED", "King Abdulaziz International Airport", "Jeddah", "Hajj and Umrah gateway"),
        ("DMM", "King Fahd International Airport", "Dammam", ""),
    ],
    "KW": [("KWI", "Kuwait International Airport", "Kuwait City", "")],
    "OM": [("MCT", "Muscat International Airport", "Muscat", "")],
    "BH": [("BAH", "Bahrain International Airport", "Manama", "")],
    "TH": [
        ("BKK", "Suvarnabhumi Airport", "Bangkok", "Main hub"),
        ("DMK", "Don Mueang International Airport", "Bangkok", "Budget carriers"),
        ("HKT", "Phuket International Airport", "Phuket", ""),
        ("CNX", "Chiang Mai International Airport", "Chiang Mai", ""),
    ],
    "SG": [("SIN", "Changi Airport", "Singapore", "")],
    "MY": [
        ("KUL", "Kuala Lumpur International Airport", "Kuala Lumpur", ""),
        ("PEN", "Penang International Airport", "Penang", ""),
    ],
    "ID": [
        ("CGK", "Soekarno-Hatta International Airport", "Jakarta", ""),
        ("DPS", "Ngurah Rai International Airport", "Bali", ""),
    ],
    "JP": [
        ("HND", "Haneda Airport", "Tokyo", ""),
        ("NRT", "Narita International Airport", "Tokyo", ""),
        ("KIX", "Kansai International Airport", "Osaka", ""),
    ],
    "KR": [
        ("ICN", "Incheon International Airport", "Seoul", ""),
        ("GMP", "Gimpo International Airport", "Seoul", ""),
    ],
    "CN": [
        ("PEK", "Beijing Capital International Airport", "Beijing", ""),
        ("PKX", "Beijing Daxing International Airport", "Beijing", ""),
        ("PVG", "Shanghai Pudong International Airport", "Shanghai", ""),
        ("CAN", "Guangzhou Baiyun International Airport", "Guangzhou", ""),
    ],
    "TR": [
        ("IST", "Istanbul Airport", "Istanbul", "Main hub"),
        ("SAW", "Sabiha Gokcen International Airport", "Istanbul", ""),
        ("AYT", "Antalya Airport", "Antalya", ""),
    ],
    "FR": [
        ("CDG", "Charles de Gaulle Airport", "Paris", ""),
        ("ORY", "Orly Airport", "Paris", ""),
        ("NCE", "Nice Cote d'Azur Airport", "Nice", ""),
    ],
    "DE": [
        ("FRA", "Frankfurt Airport", "Frankfurt", ""),
        ("MUC", "Munich Airport", "Munich", ""),
        ("BER", "Berlin Brandenburg Airport", "Berlin", ""),
    ],
    "AU": [
        ("SYD", "Sydney Kingsford Smith Airport", "Sydney", ""),
        ("MEL", "Melbourne Airport", "Melbourne", ""),
        ("BNE", "Brisbane Airport", "Brisbane", ""),
    ],
    "CA": [
        ("YYZ", "Toronto Pearson International Airport", "Toronto", ""),
        ("YVR", "Vancouver International Airport", "Vancouver", ""),
        ("YUL", "Montreal-Trudeau International Airport", "Montreal", ""),
    ],
    "BD": [("DAC", "Hazrat Shahjalal International Airport", "Dhaka", "")],
    "LK": [("CMB", "Bandaranaike International Airport", "Colombo", "")],
    "MV": [("MLE", "Velana International Airport", "Male", "")],
    "BT": [("PBH", "Paro International Airport", "Paro", "Bhutan's only international airport")],
    "PK": [
        ("KHI", "Jinnah International Airport", "Karachi", ""),
        ("LHE", "Allama Iqbal International Airport", "Lahore", ""),
        ("ISB", "Islamabad International Airport", "Islamabad", ""),
    ],
    "EG": [("CAI", "Cairo International Airport", "Cairo", "")],
}

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 71: "Light snow", 73: "Snow",
    75: "Heavy snow", 80: "Rain showers", 81: "Rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


# ==========================================================================
# HTTP helpers
# ==========================================================================
def _get_json(url: str, params: Optional[dict] = None, check_status: bool = True) -> Any:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if check_status:
            r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        raise RuntimeError("The request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP {e.response.status_code} error from {url}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error: {e}")
    except ValueError:
        raise RuntimeError("The service returned an invalid response.")


def _aviation(endpoint: str, params: Optional[dict] = None) -> dict:
    if not AVIATIONSTACK_KEY:
        raise RuntimeError("AVIATIONSTACK_API_KEY is not set in your .env file.")
    query = {"access_key": AVIATIONSTACK_KEY}
    query.update({k: v for k, v in (params or {}).items() if v not in (None, "")})
    data = _get_json(f"{AVIATIONSTACK_URL}/{endpoint}", query, check_status=False)
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        raise RuntimeError(f"Aviationstack error ({err.get('code', 'unknown')}): {err.get('message', '')}")
    return data


# ==========================================================================
# Formatting helpers
# ==========================================================================
def _fmt_time(value: Optional[str]) -> str:
    if not value:
        return "N/A"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return value


def _fmt_delay(minutes: Optional[int]) -> str:
    if not minutes:
        return "On time"
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m}m delay" if h else f"{m}m delay"


def _format_flight(f: dict) -> str:
    dep, arr = f.get("departure") or {}, f.get("arrival") or {}
    airline, info = f.get("airline") or {}, f.get("flight") or {}
    return (
        f"Flight:  {info.get('iata') or info.get('icao') or 'N/A'}\n"
        f"Airline: {airline.get('name') or 'N/A'}\n"
        f"Status:  {(f.get('flight_status') or 'unknown').title()}\n"
        f"Date:    {f.get('flight_date') or 'N/A'}\n"
        f"DEPARTURE\n"
        f"  Airport:   {dep.get('airport') or 'N/A'} ({dep.get('iata') or '?'})\n"
        f"  Terminal:  {dep.get('terminal') or 'N/A'} | Gate: {dep.get('gate') or 'N/A'}\n"
        f"  Scheduled: {_fmt_time(dep.get('scheduled'))}\n"
        f"  Estimated: {_fmt_time(dep.get('estimated'))}\n"
        f"  Delay:     {_fmt_delay(dep.get('delay'))}\n"
        f"ARRIVAL\n"
        f"  Airport:   {arr.get('airport') or 'N/A'} ({arr.get('iata') or '?'})\n"
        f"  Terminal:  {arr.get('terminal') or 'N/A'} | Gate: {arr.get('gate') or 'N/A'} "
        f"| Baggage: {arr.get('baggage') or 'N/A'}\n"
        f"  Scheduled: {_fmt_time(arr.get('scheduled'))}\n"
        f"  Estimated: {_fmt_time(arr.get('estimated'))}\n"
        f"  Delay:     {_fmt_delay(arr.get('delay'))}"
    )


def _format_flights(flights: list) -> str:
    return ("\n" + "-" * 40 + "\n").join(_format_flight(f) for f in flights)


def _format_airport(a: dict) -> str:
    return (
        f"{a.get('airport_name', 'N/A')}\n"
        f"  IATA: {a.get('iata_code', 'N/A')} | ICAO: {a.get('icao_code', 'N/A')}\n"
        f"  Country: {a.get('country_name', 'N/A')} ({a.get('country_iso2', '?')})\n"
        f"  Timezone: {a.get('timezone', 'N/A')} (GMT {a.get('gmt', 'N/A')})\n"
        f"  Coordinates: {a.get('latitude', 'N/A')}, {a.get('longitude', 'N/A')}"
    )


def _format_airline(a: dict) -> str:
    return (
        f"{a.get('airline_name', 'N/A')}\n"
        f"  IATA: {a.get('iata_code', 'N/A')} | ICAO: {a.get('icao_code', 'N/A')}\n"
        f"  Country: {a.get('country_name', 'N/A')} | Hub: {a.get('hub_code') or 'N/A'}\n"
        f"  Fleet size: {a.get('fleet_size') or 'N/A'} | Founded: {a.get('date_founded') or 'N/A'}\n"
        f"  Status: {a.get('status') or 'N/A'} | Type: {a.get('type') or 'N/A'}"
    )


# ==========================================================================
# Country helpers
# ==========================================================================
def _fetch_country(query: str) -> Optional[dict]:
    q = query.strip()
    q = COUNTRY_ALIASES.get(q.lower(), q)
    attempts = []
    if q.isalpha() and len(q) in (2, 3):
        attempts.append((f"{RESTCOUNTRIES}/alpha/{q}", None))
    attempts.append((f"{RESTCOUNTRIES}/name/{q}", {"fullText": "true"}))
    attempts.append((f"{RESTCOUNTRIES}/name/{q}", None))

    for url, params in attempts:
        try:
            data = _get_json(url, params)
        except RuntimeError:
            continue
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and "name" in data:
            return data
    return None


def _border_names(codes: list) -> list:
    if not codes:
        return []
    try:
        data = _get_json(f"{RESTCOUNTRIES}/alpha", {"codes": ",".join(codes), "fields": "name,cca3"})
        return sorted(d["name"]["common"] for d in data)
    except (RuntimeError, KeyError, TypeError):
        return codes


# ==========================================================================
# TOOLS - general
# ==========================================================================
@tool
def get_today() -> str:
    """Get today's date and weekday. Use this before planning trips or checking dates."""
    return datetime.now().strftime("Today is %A, %d %B %Y.")


# ==========================================================================
# TOOLS - countries and airports
# ==========================================================================
@tool
def get_country_info(country: str) -> str:
    """Get detailed info about a country: official name, alternate names, ISO codes,
    capital, region, languages, currency, neighbours, time zones, calling code,
    regional blocs (GCC, SAARC, BIMSTEC, ASEAN) and driving side.

    Args:
        country: Country name or code, e.g. 'Nepal', 'UAE', 'JP'.
    """
    try:
        c = _fetch_country(country)
    except RuntimeError as e:
        return f"Error: {e}"
    if not c:
        return f"Country '{country}' not found."

    cca2 = c.get("cca2", "")
    languages = ", ".join((c.get("languages") or {}).values()) or "N/A"
    currencies = ", ".join(
        f"{v.get('name')} ({code}, {v.get('symbol', '')})"
        for code, v in (c.get("currencies") or {}).items()
    ) or "N/A"
    idd = c.get("idd") or {}
    suffixes = idd.get("suffixes") or []
    calling = idd.get("root", "") + (suffixes[0] if len(suffixes) == 1 else "")
    blocs = [b for b, members in BLOCS.items() if cca2 in members]
    borders = _border_names(c.get("borders") or [])
    alt = [a for a in (c.get("altSpellings") or []) if a != cca2][:6]

    return (
        f"{c['name']['common']} ({c['name'].get('official', '')})\n"
        f"  Codes: {cca2} / {c.get('cca3', '')}\n"
        f"  Also known as: {', '.join(alt) or 'N/A'}\n"
        f"  Capital: {', '.join(c.get('capital') or ['N/A'])}\n"
        f"  Region: {c.get('region', 'N/A')} - {c.get('subregion', 'N/A')}\n"
        f"  Population: {c.get('population', 0):,} | Area: {c.get('area', 0):,.0f} km2\n"
        f"  Languages: {languages}\n"
        f"  Currency: {currencies}\n"
        f"  Calling code: {calling or 'N/A'}\n"
        f"  Time zones: {', '.join(c.get('timezones') or ['N/A'])}\n"
        f"  Neighbouring countries: {', '.join(borders) or 'None (no land borders)'}\n"
        f"  Landlocked: {c.get('landlocked', 'N/A')}\n"
        f"  Regional blocs: {', '.join(blocs) or 'None in built-in list'}\n"
        f"  Drives on the: {(c.get('car') or {}).get('side', 'N/A')}\n"
        f"  Map: {(c.get('maps') or {}).get('googleMaps', 'N/A')}"
    )


@tool
def get_main_airports(country: str) -> str:
    """List the main airports of a country (IATA code, name, city, notes).
    Use this to pick departure and arrival airports for a trip.

    Args:
        country: Country name or code, e.g. 'Nepal', 'UAE'.
    """
    try:
        c = _fetch_country(country)
    except RuntimeError as e:
        return f"Error: {e}"
    if not c:
        return f"Country '{country}' not found."

    name, code = c["name"]["common"], c.get("cca2")
    airports = MAIN_AIRPORTS.get(code)
    if not airports:
        return (
            f"No built-in airport list for {name}. "
            f"Use search_airports with a city or airport name in {name} to look them up."
        )

    lines = [f"Main airports in {name}:"]
    for iata, ap_name, city, note in airports:
        lines.append(f"  {iata} - {ap_name}, {city}" + (f" ({note})" if note else ""))
    return "\n".join(lines)


@tool
def search_airports(search: str) -> str:
    """Look up airport details (codes, timezone, coordinates) by name, city or code.

    Args:
        search: e.g. 'Heathrow', 'DXB', 'Pokhara'.
    """
    try:
        data = _aviation("airports", {"search": search.strip(), "limit": 5})
    except RuntimeError as e:
        return f"Error: {e}"
    airports = data.get("data", [])
    if not airports:
        return f"No airports found for '{search}'."
    return "\n\n".join(_format_airport(a) for a in airports)


# ==========================================================================
# TOOLS - flights
# ==========================================================================
@tool
def get_flight_status(flight_iata: str) -> str:
    """Get live status, airline, airports, terminal, gate, schedule and delay for a flight.

    Args:
        flight_iata: IATA flight code, e.g. 'EK203', 'RA205'.
    """
    try:
        data = _aviation("flights", {"flight_iata": flight_iata.strip().upper(), "limit": 3})
    except RuntimeError as e:
        return f"Error: {e}"
    flights = data.get("data", [])
    if not flights:
        return f"No flight found for {flight_iata.upper()}."
    return _format_flights(flights)


@tool
def search_flights_by_route(
    departure_iata: str,
    arrival_iata: str,
    flight_status: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Find real-time/recent flights between two airports, with status, terminal,
    gate, schedule and delay. Note: this is live data, not a fare or future schedule.

    Args:
        departure_iata: e.g. 'KTM'.
        arrival_iata: e.g. 'DXB'.
        flight_status: Optional: scheduled, active, landed, cancelled, incident, diverted.
        limit: Number of flights to return (1-100).
    """
    if flight_status and flight_status.lower() not in VALID_STATUSES:
        return f"Invalid flight_status. Choose from: {', '.join(sorted(VALID_STATUSES))}."
    try:
        data = _aviation(
            "flights",
            {
                "dep_iata": departure_iata.strip().upper(),
                "arr_iata": arrival_iata.strip().upper(),
                "flight_status": flight_status.lower() if flight_status else None,
                "limit": max(1, min(limit, 100)),
            },
        )
    except RuntimeError as e:
        return f"Error: {e}"
    flights = data.get("data", [])
    if not flights:
        return f"No flights found from {departure_iata.upper()} to {arrival_iata.upper()}."
    return f"Found {len(flights)} flight(s):\n\n" + _format_flights(flights)


@tool
def list_airlines_on_route(departure_iata: str, arrival_iata: str) -> str:
    """Show which airlines fly a route, with their flight numbers and typical
    departure times (airport local time). Great for trip planning.

    Args:
        departure_iata: e.g. 'KTM'.
        arrival_iata: e.g. 'DXB'.
    """
    try:
        data = _aviation(
            "flights",
            {
                "dep_iata": departure_iata.strip().upper(),
                "arr_iata": arrival_iata.strip().upper(),
                "limit": 100,
            },
        )
    except RuntimeError as e:
        return f"Error: {e}"

    flights = data.get("data", [])
    if not flights:
        return f"No flights found from {departure_iata.upper()} to {arrival_iata.upper()}."

    airlines: dict = {}
    for f in flights:
        name = (f.get("airline") or {}).get("name") or "Unknown"
        number = (f.get("flight") or {}).get("iata")
        sched = (f.get("departure") or {}).get("scheduled")
        entry = airlines.setdefault(name, {"flights": set(), "times": set()})
        if number:
            entry["flights"].add(number)
        if sched and len(sched) >= 16:
            entry["times"].add(sched[11:16])

    lines = [f"Airlines seen on {departure_iata.upper()} -> {arrival_iata.upper()}:"]
    for name, e in sorted(airlines.items()):
        lines.append(
            f"  - {name}: flights {', '.join(sorted(e['flights'])) or 'N/A'} | "
            f"typical departures {', '.join(sorted(e['times'])) or 'N/A'}"
        )
    lines.append("Based on recent flight records; check the airline for exact schedules and fares.")
    return "\n".join(lines)


@tool
def get_airport_board(airport_iata: str, direction: str = "departures", limit: int = 10) -> str:
    """Show the departures or arrivals board of an airport.

    Args:
        airport_iata: e.g. 'DXB'.
        direction: 'departures' or 'arrivals'.
        limit: Number of flights to return (1-100).
    """
    key = "arr_iata" if direction.lower().startswith("arr") else "dep_iata"
    try:
        data = _aviation("flights", {key: airport_iata.strip().upper(), "limit": max(1, min(limit, 100))})
    except RuntimeError as e:
        return f"Error: {e}"
    flights = data.get("data", [])
    if not flights:
        return f"No {direction} found for {airport_iata.upper()}."
    return f"{direction.title()} at {airport_iata.upper()}:\n\n" + _format_flights(flights)


@tool
def get_airline_info(search: str) -> str:
    """Look up airline details (codes, hub, fleet size, founded) by name or code.

    Args:
        search: e.g. 'Emirates', 'Buddha Air', 'QR'.
    """
    try:
        data = _aviation("airlines", {"search": search.strip(), "limit": 5})
    except RuntimeError as e:
        return f"Error: {e}"
    airlines = data.get("data", [])
    if not airlines:
        return f"No airlines found for '{search}'."
    return "\n\n".join(_format_airline(a) for a in airlines)


# ==========================================================================
# TOOLS - weather, currency, budget
# ==========================================================================
@tool
def get_weather_forecast(city: str, days: int = 7) -> str:
    """Get a daily weather forecast (up to 16 days) for a city.

    Args:
        city: City name only, e.g. 'Dubai', 'Kathmandu'.
        days: Number of forecast days (1-16).
    """
    days = max(1, min(days, 16))
    try:
        geo = _get_json(OPEN_METEO_GEO, {"name": city.split(",")[0].strip(), "count": 1})
        places = geo.get("results") or []
        if not places:
            return f"Could not find a location called '{city}'."
        p = places[0]
        fc = _get_json(
            OPEN_METEO_FORECAST,
            {
                "latitude": p["latitude"],
                "longitude": p["longitude"],
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": days,
            },
        )
    except RuntimeError as e:
        return f"Error: {e}"

    d = fc["daily"]
    lines = [f"{days}-day forecast for {p['name']}, {p.get('country', '')}:"]
    for i, day in enumerate(d["time"]):
        lines.append(
            f"  {day}: {WMO_CODES.get(d['weather_code'][i], 'N/A')}, "
            f"{d['temperature_2m_min'][i]} to {d['temperature_2m_max'][i]} C, "
            f"rain {d['precipitation_sum'][i]} mm"
        )
    return "\n".join(lines)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert money between currencies using current exchange rates.

    Args:
        amount: Amount to convert.
        from_currency: ISO code, e.g. 'USD', 'NPR'.
        to_currency: ISO code, e.g. 'AED'.
    """
    src, dst = from_currency.strip().upper(), to_currency.strip().upper()
    try:
        data = _get_json(f"{FX_URL}/{src}")
    except RuntimeError as e:
        return f"Error: {e}"
    if data.get("result") != "success":
        return f"Could not fetch rates for {src}."
    rate = (data.get("rates") or {}).get(dst)
    if rate is None:
        return f"Unknown currency code: {dst}"
    return (
        f"{amount:,.2f} {src} = {amount * rate:,.2f} {dst} "
        f"(1 {src} = {rate:.4f} {dst}, updated {data.get('time_last_update_utc', 'recently')})"
    )


@tool
def estimate_trip_budget(
    days: int,
    travelers: int = 1,
    flights_total: float = 0,
    hotel_per_night: float = 0,
    food_per_person_per_day: float = 0,
    transport_per_person_per_day: float = 0,
    activities_per_person_per_day: float = 0,
    other_total: float = 0,
    currency: str = "USD",
) -> str:
    """Calculate a trip budget exactly. flights_total and other_total cover ALL
    travelers; hotel_per_night is the total for the room(s) per night; the other
    daily costs are per person.

    Args:
        days: Number of trip days.
        travelers: Number of people.
        flights_total: Total flight cost for everyone.
        hotel_per_night: Total hotel cost per night.
        food_per_person_per_day: Food cost per person per day.
        transport_per_person_per_day: Local transport per person per day.
        activities_per_person_per_day: Tickets/activities per person per day.
        other_total: Visa, insurance, SIM, shopping, etc. for everyone.
        currency: Currency label for the output.
    """
    nights = max(days - 1, 0)
    hotel = hotel_per_night * nights
    food = food_per_person_per_day * days * travelers
    transport = transport_per_person_per_day * days * travelers
    activities = activities_per_person_per_day * days * travelers
    total = flights_total + hotel + food + transport + activities + other_total
    return (
        f"Budget for {travelers} traveler(s), {days} days / {nights} nights ({currency}):\n"
        f"  Flights:    {flights_total:,.2f}\n"
        f"  Hotel:      {hotel:,.2f}\n"
        f"  Food:       {food:,.2f}\n"
        f"  Transport:  {transport:,.2f}\n"
        f"  Activities: {activities:,.2f}\n"
        f"  Other:      {other_total:,.2f}\n"
        f"  TOTAL:      {total:,.2f}  (about {total / max(travelers, 1):,.2f} per person)"
    )


# --------------------------------------------------------------------------
# Plug in your own tools here (e.g. your existing web search tool).
# Example:
#     from my_search_tools import search_web
#     EXTRA_TOOLS = [search_web]
# --------------------------------------------------------------------------
EXTRA_TOOLS: list = []

ALL_TOOLS = [
    get_today,
    get_country_info,
    get_main_airports,
    search_airports,
    get_flight_status,
    search_flights_by_route,
    list_airlines_on_route,
    get_airport_board,
    get_airline_info,
    get_weather_forecast,
    convert_currency,
    estimate_trip_budget,
] + EXTRA_TOOLS


# ==========================================================================
# Agent
# ==========================================================================
SYSTEM_PROMPT = """You are a helpful, accurate travel assistant. You can plan trips, explain
countries, find airports, check flights, and handle weather, currency and budgets using your tools.

HOW TO WORK
- Use tools for facts. Never invent flight numbers, prices, gates, schedules or visa rules.
- Call get_today first when dates matter.
- If something is missing (dates, budget, travelers), make a sensible assumption, state it in
  one line, and continue. Only ask a question if you truly cannot proceed.

TRIP PLAN WORKFLOW (e.g. "7 day trip from Nepal to UAE")
1. get_country_info for origin and destination (currency, language, time zone, neighbours, blocs).
2. get_main_airports for both countries and pick the best airports.
3. list_airlines_on_route and search_flights_by_route for the main route.
4. get_weather_forecast for the destination cities.
5. If a web search tool is available, use it for visa rules, top attractions, hotel areas,
   local transport, safety advisories and typical prices. If no search tool is available,
   say these details come from general knowledge and should be verified.
6. convert_currency for the budget currency, then estimate_trip_budget for exact totals.
7. Write the answer.

ANSWER FORMAT FOR TRIP PLANS
- Short trip overview (route, airports, airlines seen, flight duration if known)
- Visa and entry notes (say to verify on the official government site)
- Day-by-day itinerary (morning / afternoon / evening, with area names)
- Weather and packing tips
- Budget table (budget / mid-range) with the currency converted
- Practical tips (SIM, transport, tipping, customs, local etiquette, emergency numbers)

HONESTY
- Flight data is live/recent only. It does not include fares or far-future schedules, so tell the
  user to check the airline or a booking site for prices and exact times.
- Label estimates as estimates. If a tool fails, say so and continue with what you have.
"""
from langchain_google_genai import ChatGoogleGenerativeAI

def build_agent():
    from langchain.chat_models import init_chat_model

    model_name = os.getenv("TRAVEL_AGENT_MODEL", "anthropic:claude-sonnet-4-6")
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

    try:  # LangChain 1.x
        from langchain.agents import create_agent

        return create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    except ImportError:  # older versions
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    )

