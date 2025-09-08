from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import torch
import torch.nn as nn
from torchaudio import transforms
import torch.nn.functional as F
import io
import soundfile as sf


class CheckAudio(nn.Module):
    def __init__(self, num_classes=35):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 35),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = torch.load('labels.pth')
model = CheckAudio()
model.load_state_dict(torch.load('audio_model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64,
)

max_len = 100


def change_audio_format(waveform, sample_rate):
    if sample_rate != 16000:
        new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = new_sr(torch.tensor(waveform))

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]

    elif spec.shape[1] < max_len:
        count_diff = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_diff))

    return spec


check_audio = FastAPI(title='Audio')


@check_audio.post('/predict/')
async def predict(file: UploadFile = File(..., )):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Data not found')

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.tensor(wf).T

        spec = change_audio_format(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_idx = torch.argmax(y_pred, dim=1).item()
            pred_class = labels[pred_idx]
            return {'Class': labels[pred_class]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error: {e}')


if __name__ == '__main__':
    uvicorn.run(check_audio, host='127.0.0.1', port=8000)
