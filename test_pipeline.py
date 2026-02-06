import io
from PIL import Image
from fastapi.testclient import TestClient
from api import app
from utils import haversine_distance

client = TestClient(app)

def create_dummy_image(color="red"):
    img = Image.new("RGB", (224, 224), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_haversine():
    # Distance between New York and London should be roughly 5570 km
    ny = (40.7128, -74.0060)
    lon = (51.5074, -0.1278)
    dist = haversine_distance(*ny, *lon)
    assert 5500000 < dist < 5600000 # in meters

def test_api_verification():
    img_before = create_dummy_image("black") # Damaged usually dark?
    img_after = create_dummy_image("gray")   # Repaired usually gray concrete?

    # 1. Test Same Location (Success Case)
    payload = {
        "lat_before": "12.9716",
        "lon_before": "77.5946",
        "lat_after": "12.9716",
        "lon_after": "77.5946",
    }
    files = {
        "image_before": ("before.jpg", img_before, "image/jpeg"),
        "image_after": ("after.jpg", img_after, "image/jpeg"),
    }
    
    response = client.post("/verify", data=payload, files=files)
    assert response.status_code == 200
    data = response.json()
    print("Success Response:", data)
    assert data["same_location"] is True
    assert "confidence_score" in data

    # 2. Test Different Location (Failure Case)
    payload_diff = {
        "lat_before": "12.9716",
        "lon_before": "77.5946",
        "lat_after": "13.0000", # Different lat
        "lon_after": "77.5946",
    }
    # Re-create streams because they were consumed
    img_before = create_dummy_image("black")
    img_after = create_dummy_image("gray")
    files_diff = {
        "image_before": ("before.jpg", img_before, "image/jpeg"),
        "image_after": ("after.jpg", img_after, "image/jpeg"),
    }

    response = client.post("/verify", data=payload_diff, files=files_diff)
    assert response.status_code == 200
    data = response.json()
    print("Diff Loc Response:", data)
    assert data["same_location"] is False
    assert data["confidence_score"] == 0.0

if __name__ == "__main__":
    test_haversine()
    test_api_verification()
    print("All tests passed! ✅")
