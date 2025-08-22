from torch.utils.data import DataLoader
from torchvision import transforms

transform = {
    "train" :       transforms.Compose([
                    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "validation" :  transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test" :        transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "standard" :    transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

def get_transform(dataset):
    return transform[dataset]

def initialize_loader(dataset, tf, B, shuffle):
    dataset.dataset.transform = tf
    loader = DataLoader(dataset, batch_size=B, shuffle=shuffle)
    return loader