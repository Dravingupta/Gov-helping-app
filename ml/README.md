# Road Repair Verification System

This system verifies if a pothole has been repaired by comparing two images ("Before" and "After") and checking their location.

## How It Works (Simple Terms)

The system makes decision based on **three checks**:

1.  **📍 Location Check (GPS)**
    *   We compare the latitude and longitude of both photos.
    *   If they are more than **10 meters** apart, we immediately say "Different Location" (0% confidence).
    *   *Why?* You can't repair a pothole in a different city!

2.  **🖼️ Image Similarity Check (AI)**
    *   We use a smart AI (ResNet50) that converts images into "numbers" (embeddings).
    *   We compare these numbers to see if the surronding background (road, trees, sidewalk) looks similar.
    *   *Why?* Even if the hole is filled, the surrounding road should look the same.

3.  **🔧 Repair Detection (AI)**
    *   We use another specialized AI (MobileNetV2) trained to look at a single image and answer: "Is this road damaged or repaired?"
    *   *Why?* We need to confirm the "After" image actually shows a smooth road.

## The Confidence Score

We calculate a final score (0-100%) like this:
*   **30%** comes from being at the exact same location.
*   **30%** comes from the images looking similar (background match).
*   **40%** comes from the AI confirming the repair looks good.

## How to Run

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the API**:
    ```bash
    python api.py
    ```

3.  **Use the API**:
    Send a POST request to `http://localhost:8000/verify` with:
    *   `lat_before`, `lon_before`, `lat_after`, `lon_after`
    *   `image_before` (file), `image_after` (file)
# Gov-helping-app
