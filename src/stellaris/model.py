"""Two-view CNN for exoplanet candidate classification.

Architecture follows Shallue & Vanderburg (2018), "Identifying Exoplanets with
Deep Learning". Two parallel 1D conv stacks process the global
view (2001 bins, full orbit) and local view (201 bins, transit close-up),
joined by a fully-connected head that outputs a single logit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        return x

class GlobalBranch(nn.Module):
    """
    Processes the 2001 bin global view.
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x.flatten(start_dim=1)

class LocalBranch(nn.Module):
    """
    Processes the 201 bin local view.
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(1, 16),
            ConvBlock(16, 32),
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x.flatten(start_dim=1)

class StellarisNetwork(nn.Module):
    """
    The two-view classifier.

    Output is a single logit (pre-sigmoid). Uses BCEWithLogitsLoss in training.
    Apply torch.sigmoid() at inference to get a probability in [0, 1].
    """
    def __init__(self):
        super().__init__()
        self.global_branch = GlobalBranch()
        self.local_branch = LocalBranch()

    # Determine flatten sizes by running a dummy pass at init time.
        with torch.no_grad():
            g_out = self.global_branch(torch.zeros(1, 1, 2001)) # global
            l_out = self.local_branch(torch.zeros(1, 1, 201)) # local
            flatten_dim = g_out.shape[1] + l_out.shape[1]

        self.head = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )
    
    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        g = self.global_branch(global_view)
        l = self.local_branch(local_view)
        x = torch.cat([g, l], dim=1)
        return self.head(x).squeeze(-1)

