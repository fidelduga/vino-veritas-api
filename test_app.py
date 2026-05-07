import requests
import json

url = "http://127.0.0.1:5000/predict"

headers = {
    "Content-Type": "application/json"
}

data = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()

    print("Response from local server:")
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"Error sending request: {e}")
    print("Please ensure your Flask app is running in a separate terminal.")