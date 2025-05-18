import requests
import json

# Assuming your FastAPI application is running on http://localhost:8000

url = "http://localhost:8000/query/"

# Example data matching the LabelQuery model
payload = {
    "classes": ["bird", "dog"],
    "queryFORAI": "This is an optional query string",
    "inclusivo": True
}

# If queryFORAI is not needed, you can omit it or set it to None:
# payload_without_optional_query = {
#     "classes": ["cat", "dog"]
# }
# payload_with_none_optional_query = {
#     "classes": ["apple", "banana"],
#     "queryFORAI": None
# }


try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
    
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
