
import requests
from datetime import datetime

class BusStop:
    def __init__(self, stop_name):
        self.stop_name = stop_name
        self.stop_ids = {"ILC-S": 64,"Haigis":73, "ILC-N":102}
        self.response = self.get_data(stop_name)

    def get_data(self, stop_name):
        if stop_name not in self.stop_ids:
            print("Invalid stop name")
            return []

        stop_id = self.stop_ids[stop_name]

        # Get data from URL
        url = f"https://bustracker.pvta.com/InfoPoint/rest/StopDepartures/get/{stop_id}"
        network_response = requests.get(url)

        if network_response.status_code != 200:
            print("Error fetching data:", network_response.text)
            return []

        data = network_response.json()
        response = {}

        for d in data:
            if d['StopId'] == stop_id:
                departures = d
                break
        else:
            print("Stop data not found")
            return []

        if not departures or not departures.get('RouteDirections'):
            print("No route directions found")
            return []

        for route_direction in departures['RouteDirections']:
            for departure in route_direction.get('Departures', []):
                edt = departure.get('EDTLocalTime', '')
                trip_headsign = departure['Trip'].get('InternalSignDesc', '')

                # Parse the departure time string into a datetime object
                departure_time = datetime.strptime(edt, '%Y-%m-%dT%H:%M:%S')

                # Calculate the time left for the bus to depart
                time_left = departure_time - datetime.now()

                # Convert the time left to minutes
                time_left_minutes = int(time_left.total_seconds() / 60)

                # Create or update the response dictionary
                if trip_headsign not in response:
                    response[trip_headsign] = []

                response[trip_headsign].append({'edt': edt, 'time_left_minutes': time_left_minutes})

        return response
    