import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image

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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

class RuneGenerator(nn.Module):
    def __init__(self, num_styles, num_runes):
        super(RuneGenerator, self).__init__()
        self.style_emb = nn.Embedding(num_styles, 50)
        self.rune_emb = nn.Embedding(num_runes, 50)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(150, 256, 4, 1, 0, bias=False),
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
        gen_input = torch.cat((z, style_embedding, rune_embedding.unsqueeze(2).unsqueeze(3).expand(z.size(0), 50, z.size(2), z.size(3))), 1)
        return self.gen(gen_input)

class RuneDiscriminator(nn.Module):
    def __init__(self, num_styles, num_runes):
        super(RuneDiscriminator, self).__init__()
        self.style_emb = nn.Embedding(num_styles, 1 * 48 * 48)
        self.rune_emb = nn.Embedding(num_runes, 1 * 48 * 48)
        self.dis = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img, style_labels, rune_labels):
        style_embedding = self.style_emb(style_labels).view(img.size(0), 1, 48, 48)
        rune_embedding = self.rune_emb(rune_labels).view(img.size(0), 1, 48, 48)
        dis_input = torch.cat((img, style_embedding, rune_embedding), 1)
        output = self.dis(dis_input)
        return torch.sigmoid(output.view(-1, 1))

# Determine number of unique styles and runes from the dataset
num_styles = len(dataset.style_encoder.classes_)
num_runes = len(dataset.rune_encoder.classes_)

# Initialize models
generator = RuneGenerator(num_styles, num_runes).to(device)
discriminator = RuneDiscriminator(num_styles, num_runes).to(device)

# Now you can use the generator in your generation block
with torch.no_grad():
    noise = torch.randn(1, 100, 1, 1, device=device)
    style_index = 0  # Example: 0 might correspond to 'old_carved'
    rune_index = 0   # Example: 0 might correspond to 'fehu'
    style_labels = torch.tensor([style_index], device=device).long()
    rune_labels = torch.tensor([rune_index], device=device).long()
    fake = generator(noise, style_labels, rune_labels).detach().cpu()
    img = transforms.ToPILImage()(fake[0] * 0.5 + 0.5)
    img.save('generated_rune.png')

    # Retrieve translation for the generated rune
    translation = dataset.data[dataset.data['rune_encoded'] == rune_index]['translation'].iloc[0]
    print(f"Generated rune: {dataset.rune_encoder.inverse_transform([rune_index])[0]} -> English: {translation}")