import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from wConv import wConv2d

class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()     
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=True) ##--> We have replaced this convolution
        self.conv1 = wConv2d(in_channels=1, out_channels=8, kernel_size=(3,3), den=[0.75], stride=1, padding=1, groups=1, dilation=1, bias=True) ##--> with this convolution
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2, groups=1, dilation=1, bias=True) ##--> We have replaced this convolution
        self.conv2 = wConv2d(in_channels=8, out_channels=16, kernel_size=5, den=[1.5,0.75], stride=(1,1), padding=(2,2), groups=1, dilation=1, bias=True) ##--> with this convolution
        
        self.fc = nn.Linear(16 * 16 * 16, num_classes)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1)             
        x = self.fc(x)
        return x


model = SimpleModel(num_classes=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_samples = 100
batch_size = 4
num_batches = num_samples // batch_size

inputs = torch.randn(num_samples, 1, 64, 64)
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


