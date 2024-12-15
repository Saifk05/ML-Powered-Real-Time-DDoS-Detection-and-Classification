import requests
import random
import time

URL = "http://127.0.0.1:5000/detect"

def generate_random_ip():
    """Generate a random IPv4 address."""
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def simulate_traffic():
    """Simulate traffic and send it to the Flask app."""
    while True:
        data = {
            "botnet_ip": generate_random_ip(),
            "requests": random.randint(50, 200),
            "cpu_usage": round(random.uniform(10, 90), 2),
            "bytes_sent": random.randint(1000, 10000),
            "bytes_recv": random.randint(500, 8000),
        }
        try:
            response = requests.post(URL, json=data)
            if response.status_code == 200:
                print(response.json())
            else:
                print(f"Request error: {response.text}")
        except Exception as e:
            print(f"Request error: {e}")
        time.sleep(2)  # Adjust the delay as needed

if __name__ == "__main__":
    simulate_traffic()
