import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))


class ConvNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # input 3 x 48 x 48
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Output 64 x 48 x 48
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Input 64 x 48 x 48
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Output 64 x 48 x 48
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output 64 x 24 x 24

            # input 64 x 24 x 24
            # Output 128 x 24 x 24
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Input 128 x 24 x 24
            # Output 128 x 24 x 24
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output 128 x 12 x 12

            # input 128 x 12 x 12
            # Output 256 x 12 x 12
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # input 256 x 12 x 12
            # Output 256 x 12 x 12
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output 256 x 6 x 6

            nn.Flatten(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7),
        )

    def forward(self, xb):
        return self.network(xb)
