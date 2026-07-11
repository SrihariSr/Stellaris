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

    Output is a single logit (pre-sigmoid).
    Apply torch.sigmoid() at inference to get a probability in [0, 1].

    The two branch flags exist for the ablation study. With both set to True
    (the default) this is structurally identical to the original network, so
    existing checkpoints load unchanged. Setting one to False removes that
    branch entirely: the head's input width shrinks to match, because
    flatten_dim is measured by a dummy forward pass rather than hard-coded.
    """
    def __init__(self, use_global: bool = True, use_local: bool = True):
        super().__init__()
        if not (use_global or use_local):
            raise ValueError("At least one branch must be enabled.")

        self.use_global = use_global
        self.use_local = use_local

        self.global_branch = GlobalBranch() if use_global else None
        self.local_branch = LocalBranch() if use_local else None

    # Determine flatten sizes by running a dummy pass at init time.
        with torch.no_grad():
            dims = []
            if use_global:
                dims.append(self.global_branch(torch.zeros(1, 1, 2001)).shape[1])
            if use_local:
                dims.append(self.local_branch(torch.zeros(1, 1, 201)).shape[1])
            flatten_dim = sum(dims)

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
            nn.Linear(512, 1)
        )
    
    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        # Collect whichever branches are switched on, then concatenate.
        # Both on -> identical to the original network.
        # One off -> the head simply sees a narrower feature vector.
        feats = []
        if self.use_global:
            feats.append(self.global_branch(global_view))
        if self.use_local:
            feats.append(self.local_branch(local_view))
        x = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        return self.head(x).squeeze(-1)

