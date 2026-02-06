import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from utils import get_transform
import os

def train_model(data_dir="dataset", num_epochs=10, save_path="repair_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Data Augmentation & Loading
    transform = get_transform()
    
    # Check if dataset exists and has classes
    if not os.path.exists(data_dir) or not os.path.exists(os.path.join(data_dir, "before")):
        print(f"Dataset not found at {data_dir}. Please ensure 'before' and 'after' folders exist with images.")
        return

    # Assuming 'before' = Class 0 (Damaged), 'after' = Class 1 (Repaired)
    # We use ImageFolder which assigns labels alphabetically. 
    # 'after' comes before 'before' alphabetically? No. 'after' (0), 'before' (1).
    # Wait, we want Repaired=1. 
    # 'after' starts with 'a', 'before' starts with 'b'.
    # So 'after' is 0, 'before' is 1. 
    # This is opposite of what I assumed in inference (0=Damaged, 1=Repaired).
    # I should explicitly handle classes or rename folders?
    # Let's check classes.
    
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Classes found: {dataset.classes}")
    # dataset.classes will likely be ['after', 'before']
    # 'after' index 0, 'before' index 1.
    
    # We want 'Repaired' to be 1. 'after' is repaired.
    # So currently 0 is Repaired.
    # I will stick to the dataset labels and adjust my interpretation or mapping.
    # Let's just print the mapping and use it.
    
    class_names = dataset.classes
    # We will save this mapping or just hardcode if we enforce folder names.
    # Let's enforce: '0_damaged' and '1_repaired' to be safe? 
    # Or just use 'before' and 'after' and map them.
    # If classes are ['after', 'before']:
    # after (repaired) -> 0
    # before (damaged) -> 1
    # Check inference.py: predict_repair returns probs[0][1].item(). 
    # If 1 is 'before' (damaged), then inference returns probability of DAMAGE.
    # User wants 'repaired_probability'.
    # So if after=0, prob(repaired) = probs[0][0].
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model Setup (MobileNetV2)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, Acc: {100 * correct / total:.2f}%")

    # Save Model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
