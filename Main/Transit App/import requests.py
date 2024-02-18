
# postman
# co pilot
import requests 
import json


response = requests.get("https://bustracker.pvta.com/InfoPoint/rest/StopDepartures/Get/64")
print(response.status_code)
print(response.json())

