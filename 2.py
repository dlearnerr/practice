import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
z_dim = 64 
lr = 0.0002 
batch_size = 64 
epochs = 2 
 
transform = transforms.ToTensor() 
dataset = torchvision.datasets.MNIST(root="./data", transform=transform, 
download=True) 
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
shuffle=True) 
 
class Generator(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.model = nn.Sequential( 
            nn.Linear(z_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 784), 
            nn.Tanh() 
        ) 
 
    def forward(self, x): 
        return self.model(x).view(-1, 1, 28, 28) 
 
class Discriminator(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.model = nn.Sequential( 
            nn.Flatten(), 
            nn.Linear(784, 128), 
            nn.LeakyReLU(0.2), 
            nn.Linear(128, 1), 
            nn.Sigmoid() 
        ) 
 
    def forward(self, x): 
        return self.model(x) 
 
G = Generator().to(device) 
D = Discriminator().to(device) 
 
criterion = nn.BCELoss() 
opt_G = torch.optim.Adam(G.parameters(), lr=lr) 
opt_D = torch.optim.Adam(D.parameters(), lr=lr) 
 
# Training loop 
for epoch in range(epochs): 
    for real, _ in loader: 
        real = real.to(device) 
        batch = real.size(0) 
 
        # Train Discriminator 
        noise = torch.randn(batch, z_dim).to(device) 
        fake = G(noise) 
 
        loss_D = criterion(D(real), torch.ones(batch, 1).to(device)) + \ 
                 criterion(D(fake.detach()), torch.zeros(batch, 
1).to(device)) 
 
        opt_D.zero_grad() 
        loss_D.backward() 
        opt_D.step() 
 
        # Train Generator 
        loss_G = criterion(D(fake), torch.ones(batch, 1).to(device)) 
 
        opt_G.zero_grad() 
        loss_G.backward() 
        opt_G.step() 
 
    print(f"Epoch {epoch+1}: Loss D={loss_D.item():.4f}, Loss 
G={loss_G.item():.4f}") 
 
# Generate sample images 
noise = torch.randn(5, z_dim).to(device) 
fake_images = G(noise) 
 
print("\nGenerated Samples Shape:", fake_images.shape)
