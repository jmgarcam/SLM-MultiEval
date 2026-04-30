import requests
import json
from datetime import datetime, timedelta


def fetch_newspaper_news(
    API_IP,
    API_PORT,
    newspaper,
    date_str,
    rag,
    start_hour,
    start_minute,
    end_hour,
    end_minute,
    is_synthetic,
    llm_id
):
    """Fetches newspaper news from the API based on the specified parameters."""
    
    params = (
        "date="
        + date_str
        + "&rag="
        + str(rag)
        + "&shour="
        + start_hour
        + "&sminute="
        + start_minute
        + "&ehour="
        + end_hour
        + "&eminute="
        + end_minute
        + "&id_llm="
        + str(llm_id)
    )

    if is_synthetic == True:
        url = (
            str("http://")
            + API_IP
            + ":"
            + str(API_PORT)
            + "/newsLLM/"
            + newspaper
            + "?"
        )
    else:
        url = (
            str("http://")
            + API_IP
            + ":"
            + str(API_PORT)
            + "/news/"
            + newspaper
            + "?"
        )
    print(str(url)+str(params))
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an error if the HTTP code is 4xx or 5xx
        return response.json()  # Returns the response in JSON format
    except requests.exceptions.RequestException as e:
        print(f"Error calling the endpoint: {e}")
        return None


def get_newspaper_news_history(
    API_IP, FLASK_API_PORT, start_date, end_date, newspaper, rag, llm_id
):
    """Calculates the total number of real news articles within a given date range."""
    
    start = datetime.strptime(start_date, "%d/%m/%Y")
    end = datetime.strptime(end_date, "%d/%m/%Y")

    delta = timedelta(days=1)
    current_date = start

    news_counter = 0

    while current_date <= end:
        # Format the date to "dd-mm-yyyy"
        formatted_date = current_date.strftime("%d-%m-%Y")

        # Call the function with the formatted date
        entries = fetch_newspaper_news(
            API_IP,
            FLASK_API_PORT,
            newspaper,
            formatted_date,
            rag,
            "0",  # start hour
            "00",  # start minute
            "23",  # end hour
            "00",  # end minute
            False,
            llm_id
        )["items"]

        # Advance to the next day
        current_date += delta

        news_counter += len(entries)

    return news_counter
   

def get_llm_news_history(
    API_IP, LLM_API_PORT, start_date, end_date, newspaper, rag, llm_id
):
    """Aggregates the counts of synthetic news articles by feature type over a date range."""
    
    start = datetime.strptime(start_date, "%d/%m/%Y")
    end = datetime.strptime(end_date, "%d/%m/%Y")

    delta = timedelta(days=1)
    current_date = start

    news_counter = 0

    result = {"sinteticas": 0, "t1": 0, "t2": 0, "t3": 0}

    while current_date <= end:
        # Format the date to "dd-mm-yyyy"
        formatted_date = current_date.strftime("%d-%m-%Y")

        # Call the function with the formatted date
        entries = fetch_newspaper_news(
            API_IP,
            LLM_API_PORT,
            newspaper,
            formatted_date,
            rag,
            "0",  # start hour
            "00",  # start minute
            "23",  # end hour
            "00",  # end minute
            True,
            llm_id
        )["items"]

        # Advance to the next day
        current_date += delta

        for entry in entries:
            if int(entry["id_feature"]) == 1:
                result["t1"] += 1
            elif int(entry["id_feature"]) == 2:
                result["t2"] += 1
            elif int(entry["id_feature"]) == 3:
                result["t3"] += 1

        result["sinteticas"] += len(entries)

    return result
   

def get_real_news(
    API_IP, FLASK_API_PORT, start_date, end_date, newspaper, rag, llm_id
):
    """Retrieves a list of all real news articles for a specified newspaper and date range."""
    
    start = datetime.strptime(start_date, "%d/%m/%Y")
    end = datetime.strptime(end_date, "%d/%m/%Y")

    delta = timedelta(days=1)
    current_date = start

    news = []

    while current_date <= end:
        # Format the date to "dd-mm-yyyy"
        formatted_date = current_date.strftime("%d-%m-%Y")

        # Call the function with the formatted date
        entries = fetch_newspaper_news(
            API_IP,
            FLASK_API_PORT,
            newspaper,
            formatted_date,
            rag,
            "0",   # start hour
            "00",  # start minute
            "23",  # end hour
            "00",  # end minute
            False,
            llm_id
        )["items"]
        
        # Accumulate news
        news.extend(entries)

        # Advance to the next day
        current_date += delta

    return news


def get_synthetic_news(
    API_IP, LLM_API_PORT, start_date, end_date, newspaper, rag, llm_id
):
    """Retrieves a list of all synthetic news articles for a specified newspaper and date range."""
    
    start = datetime.strptime(start_date, "%d/%m/%Y")
    end = datetime.strptime(end_date, "%d/%m/%Y")

    delta = timedelta(days=1)
    current_date = start

    news = []

    while current_date <= end:
        # Format the date to "dd-mm-yyyy"
        formatted_date = current_date.strftime("%d-%m-%Y")

        # Call the function with the formatted date
        entries = fetch_newspaper_news(
            API_IP,
            LLM_API_PORT,
            newspaper,
            formatted_date,
            rag,
            "0",  # start hour
            "00",  # start minute
            "23",  # end hour
            "00",  # end minute
            True,
            llm_id
        )["items"]
        
        # Accumulate news
        news.extend(entries)

        # Advance to the next day
        current_date += delta

    return news



def get_synthetic_news_from_json(json_filename, newspaper, llm_id):
    """
    Retrieves a flat list of synthetic news articles for a specified newspaper 
    and LLM ID directly from a structured JSON file.
    """
    try:
        with open(json_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{json_filename}'.")
        return []

    # En JSON las claves siempre son strings, nos aseguramos de convertir el id
    llm_id_str = str(llm_id)

    # Verificamos que existan datos para ese periódico y modelo
    if newspaper not in data or llm_id_str not in data[newspaper]:
        print(f"Aviso: No hay datos para {newspaper} con LLM ID {llm_id} en {json_filename}.")
        return []

    news = []

    for temperatura, entries in data[newspaper][llm_id_str].items():
        news.extend(entries)

    return news



def get_real_news_from_json(json_filename, newspaper):
    """
    Retrieves a list of real news articles for a specified newspaper 
    directly from a structured JSON file.
    """
    try:
        with open(json_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{json_filename}'.")
        return []

    # Verificamos que existan datos para ese periódico
    if newspaper not in data:
        print(f"Aviso: No hay datos para el periódico '{newspaper}' en {json_filename}.")
        return []

    # Como la estructura de las reales es directa (data[newspaper] = [lista de noticias]), 
    # simplemente la devolvemos.
    return data[newspaper]