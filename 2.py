import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
z, bs, ep = 64, 64, 2

# Data
data = DataLoader(
    datasets.MNIST("./data", transform=transforms.ToTensor(), download=True),
    batch_size=bs, shuffle=True
)

# Models
G = nn.Sequential(nn.Linear(z,128), nn.ReLU(), nn.Linear(128,784), nn.Tanh()).to(device)
D = nn.Sequential(nn.Flatten(), nn.Linear(784,128), nn.LeakyReLU(0.2), nn.Linear(128,1), nn.Sigmoid()).to(device)

# Train
loss = nn.BCELoss()
optG = torch.optim.Adam(G.parameters(), 0.0002)
optD = torch.optim.Adam(D.parameters(), 0.0002)

for e in range(ep):
    for real,_ in data:
        real = real.to(device); b = real.size(0)

        z_noise = torch.randn(b, z).to(device)
        fake = G(z_noise).view(-1,1,28,28)

        # Discriminator
        lD = loss(D(real), torch.ones(b,1).to(device)) + \
             loss(D(fake.detach()), torch.zeros(b,1).to(device))
        optD.zero_grad(); lD.backward(); optD.step()

        # Generator
        lG = loss(D(fake), torch.ones(b,1).to(device))
        optG.zero_grad(); lG.backward(); optG.step()

    print(f"Epoch {e+1}: D={lD.item():.3f}, G={lG.item():.3f}")

# Output
imgs = G(torch.randn(5, z).to(device)).view(-1,1,28,28)
print("Shape:", imgs.shape)
