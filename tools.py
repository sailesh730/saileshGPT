import math
import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from database import save_memory,get_chat_history
load_dotenv()
from rag import retrive_from_rag
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
def rember_this(memory:str)->str:
    """save an importent user performence or fact into lang-trm memory 
    used this whem the user ask toyou to rember something"""
    return save_memory (
        thred_id_set=CUREENT_THRED_ID,
        memory=memory
    )
@tool 
def recall_memory(query:str)->str:
    """recall save long term memories about the user or this conversation
    """
    return get_chat_history(
        thread_id=CUREENT_THRED_ID,
        query=query
    )


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

