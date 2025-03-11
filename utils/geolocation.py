import googlemaps
from geopy.distance import geodesic

GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"

def get_user_location():
    """
    Get the user's current location (latitude, longitude).
    Replace this with actual implementation.
    """
    return (37.7749, -122.4194)  # Example: San Francisco

def find_nearby_dermatologists(user_location, radius=10):
    """
    Find dermatologists within a given radius (in km) using Google Maps API.
    """
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    places = gmaps.places_nearby(
        location=user_location,
        radius=radius * 1000,  # Convert km to meters
        type="doctor",
        keyword="dermatologist"
    )
    results = []
    for place in places.get("results", []):
        name = place.get("name")
        address = place.get("vicinity")
        location = place.get("geometry", {}).get("location")
        distance = geodesic(user_location, (location["lat"], location["lng"])).km
        results.append({
            "name": name,
            "address": address,
            "distance": round(distance, 2)
        })
    return results