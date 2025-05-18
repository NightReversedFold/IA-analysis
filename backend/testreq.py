import requests



url = "http://localhost:8000/query/"

payload = {
    "classes": ["person"],
    "queryFORAI": "pajaro",
    "inclusivo": True
}


try:
    response = requests.post(url, json=payload)
    response.raise_for_status() 
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json()) 
except requests.exceptions.HTTPError as errh:
    print(f"Http Error: {errh}")
except requests.exceptions.ConnectionError as errc:
    print(f"Error Connecting: {errc}")
except requests.exceptions.Timeout as errt:
    print(f"Timeout Error: {errt}")
except requests.exceptions.RequestException as err:
    print(f"OOps: Something Else: {err}")
