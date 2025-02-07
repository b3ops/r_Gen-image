import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# ... (Previous imports and class definitions remain unchanged)
# Check if CUDA is available for GPU acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        style_label = self.data.iloc[idx]['style_encoded']
        rune_label = self.data.iloc[idx]['rune_encoded']
        translation = self.data.iloc[idx]['translation']
        return image, (style_label, rune_label, translation)

# Define transformations for preprocessing runes
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Since your images are 48x48
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = ElderFutharkDataset('runes_dataset.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)  # Set batch_size to 1 for individual processing

# Iterate through dataset
for i, (image, labels) in enumerate(dataloader):
    _, (style_label, rune_label, translation) = image, labels
    
    # Convert back to PIL Image for display if needed
    pil_image = transforms.ToPILImage()(image[0] * 0.5 + 0.5)
    
    # Decode the rune and style labels
    rune = dataset.rune_encoder.inverse_transform([rune_label.item()])[0]
    style = dataset.style_encoder.inverse_transform([style_label.item()])[0]
    
    # Print information
    print(f"Image {i+1}:")
    print(f"  Rune: {rune}")
    print(f"  Style: {style}")
    print(f"  Translation: {translation}")
    
    # If you need to save or display images, uncomment the following:
    # pil_image.save(f'processed_rune_{i+1}.png')