import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class
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
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        style_label = self.data.iloc[idx]['style_encoded']
        rune_label = self.data.iloc[idx]['rune_encoded']
        translation = self.data.iloc[idx]['translation']
        return image, (style_label, rune_label, translation)

class RuneGenerator(nn.Module):
    def __init__(self, num_styles, num_runes):
        super(RuneGenerator, self).__init__()
        self.style_emb = nn.Embedding(num_styles, 50)
        self.rune_emb = nn.Embedding(num_runes, 50)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(200, 256, 4, 1, 0, bias=False),  # Changed to match input size
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, style_labels, rune_labels):
        style_embedding = self.style_emb(style_labels)
        rune_embedding = self.rune_emb(rune_labels)
        
        # Adjust the dimensions of embeddings to match z
        style_embedding = style_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, z.size(2), z.size(3))
        rune_embedding = rune_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, z.size(2), z.size(3))
        
        gen_input = torch.cat((z, style_embedding, rune_embedding), 1)
        return self.gen(gen_input)
        
# Transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = ElderFutharkDataset('runes_dataset.csv', transform=transform)

# Initialize models
num_styles = len(dataset.style_encoder.classes_)
num_runes = len(dataset.rune_encoder.classes_)
generator = RuneGenerator(num_styles, num_runes).to(device)
generator.eval()  # Evaluation mode

# Create rune to translation mapping
rune_to_translation = dataset.data.set_index('rune')['translation'].to_dict()

# Main function
def main():
    print("Enter text to translate into runes (one character at a time):")
    while True:
        char = input("Enter a character (or 'q' to quit): ").strip().lower()
        if char == 'q':
            break
        
        # Find rune for the character
        rune_name = next((rune for rune, translation in rune_to_translation.items() if translation.lower() == char), None)
        if not rune_name:
            print(f"No rune found for '{char}'. Skipping.")
            continue

        rune_index = dataset.rune_encoder.transform([rune_name])[0]

        # Generate rune image
        with torch.no_grad():
            noise = torch.randn(1, 100, 1, 1, device=device)
            style_index = 0  # Default style
            style_labels = torch.tensor([style_index], device=device).long()
            rune_labels = torch.tensor([rune_index], device=device).long()
            fake = generator(noise, style_labels, rune_labels).detach().cpu()
            img = transforms.ToPILImage()(fake[0] * 0.5 + 0.5)
            img.save(f'generated_rune_{char}.png')
            
            print(f"Generated rune for '{char}' as '{rune_name}' saved as 'generated_rune_{char}.png'")
            print(f"Translation for '{rune_name}': {rune_to_translation[rune_name]}")

if __name__ == "__main__":
    main()