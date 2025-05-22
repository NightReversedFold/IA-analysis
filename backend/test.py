import requests

url = 'http://127.0.0.1:8000/video/'
file = {'file': open('arre.mp4', 'rb')}
resp = requests.post(url=url, files=file) 
print(resp.json())