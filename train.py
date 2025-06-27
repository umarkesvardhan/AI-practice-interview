import os
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")  # ✅ Force backend that works with most .wav files
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ✅ Configuration
DATA_DIR = r"D:\AI mock interview"  # Use raw string (r"...") to avoid backslash issues
SAMPLE_RATE = 16000
BATCH_SIZE = 16
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Custom Dataset
class AudioDataset(Dataset):
    def __init__(self, directory):
        self.files = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        if not os.path.exists(directory):
            raise FileNotFoundError(f"❌ Dataset path '{directory}' does not exist.")

        speakers = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        self.label_encoder.fit(speakers)

        for speaker in speakers:
            speaker_dir = os.path.join(directory, speaker)
            for file in os.listdir(speaker_dir):
                if file.lower().endswith(".wav"):
                    self.files.append(os.path.join(speaker_dir, file))
                    self.labels.append(speaker)

        if not self.files:
            raise ValueError("❌ No .wav files found in dataset folders.")

        self.labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx] 
        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
        waveform = waveform.mean(dim=0)  # Mono

        if waveform.shape[0] < SAMPLE_RATE:
            waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.shape[0]))
        else:
            waveform = waveform[:SAMPLE_RATE]

        return waveform, self.labels[idx]

# ✅ Simple Feedforward Speaker Classification Model
class SpeakerModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeakerModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(SAMPLE_RATE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ✅ Training Function
def train():
    print("🔍 Loading dataset...")
    dataset = AudioDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(set(dataset.labels))

    print(f"🎙️ Detected {num_classes} speakers, {len(dataset)} audio samples.")
    model = SpeakerModel(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, targets = inputs.to(DEVICE), torch.tensor(targets).to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📉 Epoch {epoch+1} | Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/speaker_recognition_model.pth")
    print("✅ Model saved to models/speaker_recognition_model.pth")

# ✅ Entry Point
if __name__ == "__main__":
    train()
