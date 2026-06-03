import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# ==========================================
# BOILERPLATE: Architecture & Data
# ==========================================

class TimeConditionedMLP(nn.Module):
    """A simple network that predicts noise based on data x and time t."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim), # Input: 2D coordinates + 1D time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)      # Output: 2D noise prediction
        )

    def forward(self, x, t):
        # Concatenate spatial coordinates and time
        t = t.unsqueeze(-1).float() / 100.0 # simple time scaling
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

def get_swiss_roll(n_samples=10000):
    """Generates 2D Swiss Roll data."""
    x, _ = make_swiss_roll(n_samples, noise=0.5)
    data = np.vstack([x[:, 0], x[:, 2]]).T
    # Normalize to roughly [-1, 1]
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data, dtype=torch.float32)

# ==========================================
# THE CORE MATH: Your Playground
# ==========================================

class ToyDDPM(nn.Module):
    def __init__(self, network, n_steps=100):
        super().__init__()
        self.network = network
        self.n_steps = n_steps
        
        # Define the linear variance schedule (betas)
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        
        # Calculate alphas and alpha_bars (cumulative products)
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_bar', torch.cumprod(alphas, dim=0))

    def q_sample(self, x_0, t, noise):
        """
        [DONE] 1: The Forward Process
        Implement the forward perturbation kernel q(x_t | x_0).
        Calculate x_t given x_0, the timestep t, and the injected Gaussian noise.
        Hint: You'll need self.alphas_bar[t].
        """
        # Unsqueeze to reshape from [batch_size] to [batch_size, 1]
        # Here x_0 and noise have size [batch_size, 2], since here we have 2-d swiss roll data
        # The unsqueeze allows pytorch to broadcast the multiplication across dimensions,
        #   which it does when the rightmost dimension is 1
        sqrt_abar = torch.sqrt(self.alphas_bar[t]).unsqueeze(-1)
        sqrt_1m_abar = torch.sqrt(1 - self.alphas_bar[t]).unsqueeze(-1)
        x_t = sqrt_abar * x_0 + sqrt_1m_abar * noise
        return x_t

    def compute_loss(self, x_0):
        """
        [DONE] 2: The Objective Function
        1. Sample random timesteps 't' for the batch.
        2. Generate standard Gaussian noise.
        3. Get x_t using your q_sample function.
        4. Predict the noise using self.network(x_t, t).
        5. Return the Mean Squared Error between the true noise and predicted noise.
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_steps, size=(batch_size,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)  # standard Gaussian, same size as x_0
        x_t = self.q_sample(x_0, t, noise)
        pred_noise = self.network(x_t, t)  # predict noise using neural network
        loss = nn.functional.mse_loss(pred_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, n_samples):
        """
        [DONE] 3: The Reverse Process (Generation)
        Start from pure noise x_T, and iteratively denoise to x_0.
        Use the network to predict the noise at each step, and apply the reverse 
        Langevin dynamics update rule to step from x_t to x_{t-1}.
        """
        x_t = torch.randn((n_samples, 2), device=self.betas.device)
        
        for i in reversed(range(self.n_steps)):
            t = torch.full((n_samples,), i, device=self.betas.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.network(x_t, t)

            # Compute constants
            alpha_t = 1.0 - self.betas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(-1)
            sqrt_abar = torch.sqrt(self.alphas_bar[t]).unsqueeze(-1)
            sqrt_1m_abar = torch.sqrt(1 - self.alphas_bar[t]).unsqueeze(-1)
            beta_div_abar = self.betas[t].unsqueeze(-1) / sqrt_1m_abar

            # Compute deterministic model mean (without added noise)
            model_mean = (1 / sqrt_alpha_t) * (x_t - beta_div_abar * predicted_noise)

            # Add stochastic noise (Langevin dynamics), unless it's the final step
            if i > 0:
                z = torch.randn_like(x_t)  # Independent noise for each dimension
                sigma_t = torch.sqrt(self.betas[t]).unsqueeze(-1) # Standard deviation
                x_t = model_mean + sigma_t * z
            else:
                x_t = model_mean
            
        return x_t

# ==========================================
# BOILERPLATE: Training Loop & Execution
# ==========================================

def train_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    
    # Setup
    dataset = get_swiss_roll().to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = ToyDDPM(TimeConditionedMLP()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 201
    print("Starting training...")
    
    # Training Loop
    for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            if epoch % 50 == 0:
                print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    # Evaluation & Plotting
    print("Sampling from model...")
    samples = model.sample(2000).cpu().numpy()
    real_data = dataset[:2000].cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(real_data[:, 0], real_data[:, 1], s=5, alpha=0.5, c='blue')
    plt.title("Real Swiss Roll")
    
    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, c='red')
    plt.title("DDPM Generated")
    # plt.show()
    plt.savefig('images/swiss_rolls.png')

if __name__ == "__main__":
    train_and_plot()
    pass
