import torch
import torch.nn as nn
from torchvision import models
from utils import get_transform

class RoadVerificationModel:
    def __init__(self, model_path="repair_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_transform()
        
        # 1. Similarity Model (ResNet50 pre-trained)
        # We use the penultimate layer for embeddings
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.similarity_model = nn.Sequential(*list(resnet.children())[:-1])
        self.similarity_model.to(self.device)
        self.similarity_model.eval()

        # 2. Repair Detection Model (MobileNetV2)
        self.repair_model = models.mobilenet_v2(weights=None)
        # Modify classifier for binary classification (0: Damaged, 1: Repaired)
        self.repair_model.classifier[1] = nn.Linear(self.repair_model.last_channel, 2)
        
        try:
            self.repair_model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded repair model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: {model_path} not found. Using initialized weights (unreliable predictions).")
        
        self.repair_model.to(self.device)
        self.repair_model.eval()

    def get_embedding(self, image):
        """Extract feature vector from image."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.similarity_model(img_tensor)
        return torch.flatten(embedding).cpu()

    def compute_similarity(self, image1, image2):
        """Compute cosine similarity between two images."""
        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)
        cos = nn.CosineSimilarity(dim=0)
        return cos(emb1, emb2).item()

    def predict_repair(self, image):
        """Return probability that the image is 'Repaired'."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.repair_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            # Index 0 is 'after' (Repaired), Index 1 is 'before' (Damaged)
            return probs[0][0].item()
