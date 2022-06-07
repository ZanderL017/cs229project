import torch
import torch.nn as nn
import torch.nn.functional as F

class IGTDImages(torch.utils.data.Dataset):
    def __init__(self, image_data, labels):
        super().__init__()
        self.images = torch.from_numpy(image_data).float()
        self.labels = torch.from_numpy(labels).long()
        assert len(self.images) == len(self.labels)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        return self.images[i].unsqueeze(0), self.labels[i]

class SimpleCNN(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding="same")
        self.batch1 = nn.LayerNorm((32, 32))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
        self.batch2 = nn.LayerNorm((16, 16))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
        self.batch3 = nn.LayerNorm((8, 8))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 64, 128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_out)
        self.drop2 = nn.Dropout(0.5)
        self.out = nn.Softmax()
    
    def forward(self, x):
        x = self.pool1(self.batch1(F.relu(self.conv1(x))))
        x = self.pool2(self.batch2(F.relu(self.conv2(x))))
        x = self.pool3(self.batch3(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.drop1(self.fc1(x)))
        x = self.out(self.drop2(self.fc2(x)))
        return x

    

