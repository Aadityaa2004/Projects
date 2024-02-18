import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PVTA_BASE_URL = "https://bustracker.pvta.com/InfoPoint/rest/StopDepartures/get/"
STOP_IDS = {
    33: 64,
    # Add more stop IDs as needed
}

def get_bus_departures(bus_number):
    """Fetch bus departures from the PVTA API."""
    stop_id = STOP_IDS.get(bus_number)
    if stop_id is None:
        logger.error(f"No stop ID found for bus number {bus_number}")
        return []

    url = f"{PVTA_BASE_URL}{stop_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        departures_data = response.json()
        return departures_data
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data from {url}: {e}")
        return []

def parse_departures(departures_data):
    """Parse bus departures data."""
    departures = []
    for route_direction in departures_data:
        for departure in route_direction.get('Departures', []):
            departure_info = {
                'edt': departure.get('EDTLocalTime'),
                'trip_headsign': departure['Trip'].get('InternalSignDesc'),
                'trip_direction': departure['Trip'].get('TripDirection')
            }
            departures.append(departure_info)
    return departures

def main():
    bus_number = 33
    departures_data = get_bus_departures(bus_number)
    if not departures_data:
        logger.info("No departure data available.")
        return

    departures = parse_departures(departures_data)
    print(departure)
    for departure in departures:
        print(f"EDT: {departure['edt']}, Headsign: {departure['trip_headsign']}, Direction: {departure['trip_direction']}")

if __name__ == "__main__":
    main()
