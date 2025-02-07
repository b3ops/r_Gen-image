import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os

# Device setup (not needed for this operation, but kept for consistency)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class (slightly modified to return just the image path)
class ElderFutharkDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.style_encoder = LabelEncoder()
        self.rune_encoder = LabelEncoder()
        self.data['style_encoded'] = self.style_encoder.fit_transform(self.data['style'])
        self.data['rune_encoded'] = self.rune_encoder.fit_transform(self.data['rune'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['file_path']
        rune = self.data.iloc[idx]['rune']
        translation = self.data.iloc[idx]['translation']
        return img_path, rune, translation  # Return path, rune, and translation

# Transformations (not used for this operation but kept for potential future use)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = ElderFutharkDataset('runes_dataset.csv', transform=None)

# Create rune to translation mapping
rune_to_translation = dataset.data.set_index('rune')['translation'].to_dict()

# Create translation to rune mapping for quick lookup
translation_to_rune = {v.lower(): k for k, v in rune_to_translation.items()}

def main():
    print("Enter text to translate into runes (one character at a time):")
    while True:
        char = input("Enter a character (or 'q' to quit): ").strip().lower()
        if char == 'q':
            break
        
        rune_name = translation_to_rune.get(char, None)
        if not rune_name:
            print(f"No rune found for '{char}'. Skipping.")
            continue

        # Find all images for this rune
        matching_rows = dataset.data[dataset.data['rune'] == rune_name]
        if matching_rows.empty:
            print(f"No image found for rune '{rune_name}'. Skipping.")
            continue

        # Use the first matching image (you might want to handle multiple images differently)
        img_path = matching_rows.iloc[0]['file_path']
        image = Image.open(img_path).convert('L')  # Convert to grayscale if needed

        # Save the image
        save_path = f'generated_rune_{char}.png'
        image.save(save_path)
        print(f"Generated rune for '{char}' as '{rune_name}' saved as '{save_path}'")
        print(f"Translation for '{rune_name}': {matching_rows.iloc[0]['translation']}")

if __name__ == "__main__":
    main()