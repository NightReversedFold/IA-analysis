import requests
import argparse

url = "http://localhost:8000/query/"

# Create an ArgumentParser instance
parser = argparse.ArgumentParser(description="Send a query to the FastAPI backend.")

# Add arguments
parser.add_argument("--c", nargs="+", required=True, help="List of classes (e.g., dog sofa)")
parser.add_argument("--q", type=str, default=None, help="Query string for AI (optional)")
parser.add_argument("--i", dest='inclusivo', action='store_true', help="Set to inclusive (default)")
parser.add_argument("--n", dest='inclusivo', action='store_false', help="Set to not inclusive")
parser.set_defaults(inclusivo=True)

# Parse arguments
args = parser.parse_args()

# Print received arguments
print("Received arguments:")
print(f"  Classes: {args.c}")
print(f"  Query FOR AI: {args.q}")
print(f"  Inclusivo: {args.inclusivo}")
print("-" * 20) # Separator

payload = {
    "classes": args.c,
    "queryFORAI": args.q,
    "inclusivo": args.inclusivo
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
