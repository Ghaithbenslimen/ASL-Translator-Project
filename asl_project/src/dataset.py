from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms

CLASS_NAMES = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
               'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
               'del','nothing','space']
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}


def build_transforms(train, img_size=128):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


class ASLDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.samples = []
        root = Path(root)
        for cls_name in CLASS_NAMES:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                print(f"  [warn] missing: {cls_dir}")
                continue
            idx = CLASS_TO_IDX[cls_name]
            for p in cls_dir.iterdir():
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    self.samples.append((str(p), idx))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_loaders(data_root, img_size=128, batch_size=64, num_workers=2, val_split=0.1):
    import random
    ds_tr = ASLDataset(data_root, build_transforms(True, img_size))
    ds_va = ASLDataset(data_root, build_transforms(False, img_size))
    n = len(ds_tr)
    idx = list(range(n))
    random.Random(42).shuffle(idx)
    vn = int(n * val_split)
    tr = DataLoader(Subset(ds_tr, idx[vn:]), batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(Subset(ds_va, idx[:vn]), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return tr, va
