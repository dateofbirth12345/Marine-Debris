from __future__ import annotations

from typing import Optional, Tuple
from PIL import Image
import exifread


def _to_degrees(value):
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)


def extract_gps_from_image(image_path: str) -> Optional[Tuple[float, float]]:
    """Extract latitude and longitude from an image EXIF if present.

    Returns:
        (lat, lon) in decimal degrees or None if not available.
    """
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        gps_lat = tags.get("GPS GPSLatitude")
        gps_lat_ref = tags.get("GPS GPSLatitudeRef")
        gps_lon = tags.get("GPS GPSLongitude")
        gps_lon_ref = tags.get("GPS GPSLongitudeRef")

        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            lat = _to_degrees(gps_lat)
            if gps_lat_ref.values != "N":
                lat = -lat
            lon = _to_degrees(gps_lon)
            if gps_lon_ref.values != "E":
                lon = -lon
            return lat, lon
    except Exception:
        return None
    return None




