import math
from io import BytesIO
from PIL import Image, ImageStat, ImageFilter
import numpy as np
from torchvision import transforms

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 * 1000 # Radius of earth in meters
    return c * r

def load_image(image_data: bytes) -> Image.Image:
    """Load image from bytes and convert to RGB."""
    return Image.open(BytesIO(image_data)).convert("RGB")

def get_transform():
    """Returns the transformation pipeline for the model."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def calculate_roughness(image: Image.Image) -> float:
    """
    Estimates surface roughness using edge detection and standard deviation.
    Returns a score between 0.0 (Smooth) and 1.0 (Very Rough/Pothole).
    """
    # Convert to grayscale
    gray_img = image.convert("L")
    
    # 1. Edge Detection
    edges = gray_img.filter(ImageFilter.FIND_EDGES)
    stat_edges = ImageStat.Stat(edges)
    avg_edge_intensity = stat_edges.mean[0]
    
    # 2. Texture Variance (Standard Deviation)
    # Potholes have high contrast (dark holes vs light pavement) -> High StdDev
    # Repaired roads are uniform -> Low StdDev
    stat_var = ImageStat.Stat(gray_img)
    std_dev = stat_var.stddev[0]
    
    # Normalize scores with balanced thresholds
    # Edges: 100 seems to be the sweet spot (Road~30->0.3, Pothole~60->0.6)
    edge_score = min(avg_edge_intensity / 100.0, 1.0)
    
    # StdDev: 70 seems balanced (Road~20->0.3, Pothole~50->0.7)
    std_score = min(std_dev / 70.0, 1.0)
    
    # Combined Score (Average)
    final_score = (edge_score + std_score) / 2.0
    
    return final_score
