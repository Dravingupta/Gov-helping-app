from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import uvicorn
from utils import haversine_distance, load_image, calculate_roughness
from inference import RoadVerificationModel

app = FastAPI(title="Road Repair Verification API")

# Load model once at startup
print("Loading models... this may take a moment.")
pipeline = RoadVerificationModel()

@app.get("/")
def home():
    return {"status": "Road Repair Verification API is running 🛣️✅"}

@app.post("/verify")
async def verify_repair(
    lat_before: float = Form(...),
    lon_before: float = Form(...),
    lat_after: float = Form(...),
    lon_after: float = Form(...),
    image_before: UploadFile = File(...),
    image_after: UploadFile = File(...)
):
    # 1. Location Verification
    distance = haversine_distance(lat_before, lon_before, lat_after, lon_after)
    is_same_location = distance <= 50.0 # 50 meters tolerance
    
    if not is_same_location:
         return {
            "same_location": False,
            "repaired": False,
            "confidence_score": 0.0,
            "details": {
                "distance_meters": round(distance, 2)
            }
        }

    # 2. Process Images
    img_b = load_image(await image_before.read())
    img_a = load_image(await image_after.read())

    # 3. Image Similarity
    similarity = pipeline.compute_similarity(img_b, img_a)
    
    # CRITICAL: If images are too similar, no repair could have happened.
    # A repair visually changes the surface (Pothole -> Asphalt).
    # If Similarity > 0.95, it's virtually the same image.
    if similarity > 0.95:
         return {
            "same_location": True,
            "repaired": False,
            "confidence_score": 0.0,
            "details": {
                "distance_meters": round(distance, 2),
                "similarity_score": round(similarity, 4),
                "message": "Images are identical. A repair produces a visual change."
            }
        }
    # 4. Repair Detection (Hybrid: ML + Heuristic)
    # We need to ensure BEFORE is "Damaged" and AFTER is "Repaired"
    
    # ML Model is UNTRAINED so we set weight to 0.0 to avoid noise.
    # Rely 100% on Heuristic (Roughness) for now.
    
    # 4a. Analyze "After" Image
    ml_after_prob = pipeline.predict_repair(img_a)
    roughness_after = calculate_roughness(img_a)
    prob_after_is_repaired = 1.0 - roughness_after

    # 4b. Analyze "Before" Image
    ml_before_prob = pipeline.predict_repair(img_b)
    roughness_before = calculate_roughness(img_b)
    prob_before_is_repaired = 1.0 - roughness_before

    # 5. Logic & Confidence Score
    # Weights: GPS (0.30), Similarity (0.30), Repair Quality (0.40)
    
    gps_score = 1.0
    sim_score = min(similarity / 0.75, 1.0) 
    
    # Repair Quality Metric:
    # Requires: Before is NOT repaired (1 - prob_before) AND After IS repaired (prob_after)
    # If Before is already repaired (e.g. 0.9), term becomes (1-0.9)=0.1 -> Score drops massive.
    repair_quality_score = (1.0 - prob_before_is_repaired) * prob_after_is_repaired
    
    confidence = (gps_score * 0.30) + (sim_score * 0.30) + (repair_quality_score * 0.40)
    confidence_percent = round(confidence * 100, 2)
    
    is_repaired = confidence_percent > 70.0 

    return {
        "same_location": True,
        "repaired": is_repaired,
        "confidence_score": confidence_percent,
        "details": {
            "distance_meters": round(distance, 2),
            "similarity_score": round(similarity, 4),
            "before_repaired_prob": round(prob_before_is_repaired, 4),
            "after_repaired_prob": round(prob_after_is_repaired, 4),
            "repair_quality_score": round(repair_quality_score, 4)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
