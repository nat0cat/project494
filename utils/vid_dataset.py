import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class VideoSeqDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_len=5, frame_gap=1):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.frame_gap = frame_gap
        self.samples = []

        # iterate over video folders
        for folder in sorted(os.listdir(root_dir)):
            f_path = os.path.join(root_dir, folder)
            if not os.path.isdir(f_path):
                continue
            files = sorted(os.listdir(f_path))

            # slide a window of length seq_len across the frames
            for i in range(0, len(files) - (seq_len - 1) * frame_gap):
                frame_paths = [
                    os.path.join(f_path, files[i + j * frame_gap])
                    for j in range(seq_len)
                ]
                self.samples.append(frame_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]
        frames = [Image.open(f).convert("RGB") for f in frame_paths]
        frames = torch.stack([self.transform(f) for f in frames], dim=0)

        return frames
