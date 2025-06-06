import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from wConv import wConv3d

class SimpleModel3D(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel3D, self).__init__()
        #self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=True) ##--> We have replaced this convolution
        self.conv1 = wConv3d(in_channels=1, out_channels=8, kernel_size=3, den=[0.75], padding=1, bias=True) ##--> with this convolution
        self.pool = nn.MaxPool3d(2, 2)
        #self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=5, padding=2, bias=True) ##--> We have replaced this convolution
        self.conv2 = wConv3d(in_channels=8, out_channels=16, kernel_size=5, den=[0.25, 0.75], padding=2, bias=True) ##--> with this convolution

        self.fc = nn.Linear(16 * 8 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleModel3D(num_classes=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_samples = 100
batch_size = 4
num_batches = num_samples // batch_size

inputs = torch.randn(num_samples, 1, 32, 32, 32)
targets = torch.randint(0, 10, (num_samples,))

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i in range(num_batches):
        batch_inputs = inputs[i*batch_size:(i+1)*batch_size].to(device)
        batch_targets = targets[i*batch_size:(i+1)*batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


