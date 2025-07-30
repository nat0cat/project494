from torch.utils.data import Dataset
import os
from PIL import Image

# dataset class for sequential data
class SeqDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_gap=1):
        self.transform = transform  # transformation applied
        self.samples = []           # samples of data

        # open data folders from root directory
        folders = os.listdir(root_dir)

        # iterate over folders
        for folder in folders:
            # find files in the folder
            f_path = os.path.join(root_dir, folder)
            files = sorted(os.listdir(f_path))

            # iterate over groups of image frames
            for i in range(frame_gap, len(files) - frame_gap):
                # structure the data in each group
                self.samples.append({
                    "src_prev": os.path.join(f_path, files[i - frame_gap]),
                    "target": os.path.join(f_path, files[i]),
                    "src_next": os.path.join(f_path, files[i + frame_gap]),
                })

    # returns the number of samples
    def __len__(self):
        return len(self.samples)

    # get items from the dataset
    def __getitem__(self, idx):
        # locate sample at index
        sample = self.samples[idx]

        # reformat data
        tgt = Image.open(sample["target"]).convert("RGB")
        src1 = Image.open(sample["src_prev"]).convert("RGB")
        src2 = Image.open(sample["src_next"]).convert("RGB")

        # apply transformations if any
        if self.transform:
            tgt = self.transform(tgt)
            src1 = self.transform(src1)
            src2 = self.transform(src2)

        # return tuple of data
        return tgt, [src1, src2]