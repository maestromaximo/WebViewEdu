import requests

url = "YOUR_API_URL/detect"
files = {'file': open('/path/to/your/image.png', 'rb')}
response = requests.post(url, files=files)

print(response.json())
