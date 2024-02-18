from Linkdrawing import *

ilc_s = BusStop('ILC-S')
bus_data= ilc_s.response

# haigis = BusStop('Haigis')
# bus_data = haigis.response

# Printing the grouped and categorized data
for headsign, departures in bus_data.items():
    print(f"Bus Headsign: {headsign}")
    # for departure in departures:
    #     print(f"Departure Time: {departure['edt']}, Time Left: {departure['time_left_minutes']} minutes")
    print(departures[0]["time_left_minutes"],"mins left")

