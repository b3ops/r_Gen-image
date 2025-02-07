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
        rune = self.data.iloc[idx]['rune']
        translation = self.data.iloc[idx]['translation']
        return img_path, rune, translation

# Transformations (not used here but kept for consistency)
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
    print("Enter text to translate into runes (whole word):")
    while True:
        word = input("Enter a word (or 'q' to quit): ").strip().lower()
        if word == 'q':
            break

        output_data = []
        images = []
        
        for char in word:
            rune_name = translation_to_rune.get(char, None)
            if rune_name:
                matching_rows = dataset.data[dataset.data['rune'] == rune_name]
                if not matching_rows.empty:
                    img_path = matching_rows.iloc[0]['file_path']
                    image = Image.open(img_path).convert('L')  # Convert to grayscale if needed
                    save_path = f'generated_rune_{char}.png'
                    image.save(save_path)
                    images.append(image)
                    
                    # Record the mapping
                    output_data.append({
                        'character': char,
                        'rune': rune_name,
                        'image_path': os.path.abspath(save_path)
                    })
                else:
                    print(f"No image found for rune '{rune_name}' for character '{char}'.")
            else:
                print(f"No rune found for '{char}'.")

        if output_data:
            # Combine images
            width = len(images) * 48
            height = 48
            combined_image = Image.new('L', (width, height))
            
            for i, img in enumerate(images):
                combined_image.paste(img, (i * 48, 0))
            
            combined_save_path = f'combined_runes_{word}.png'
            combined_image.save(combined_save_path)

            # Create a DataFrame with the mappings
            output_df = pd.DataFrame(output_data)
            # Save the DataFrame to a new CSV file
            output_csv_path = f'output_runes_{word}.csv'
            output_df.to_csv(output_csv_path, index=False)
            
            print(f"Generated and combined runes for '{word}' into {combined_save_path}")
            print(f"Mappings saved to {output_csv_path}")
            for entry in output_data:
                print(f"Character '{entry['character']}' -> Rune '{entry['rune']}' saved as '{entry['image_path']}'")

if __name__ == "__main__":
    main()