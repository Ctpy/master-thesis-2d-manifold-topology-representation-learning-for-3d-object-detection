from typing import Any, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.utils import index_points
from torch_geometric.nn.pool import knn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from pytorch_lightning.utilities.types import STEP_OUTPUT



class FoldingNet(pl.LightningModule):
    """
    FoldingNet.
    """

    def __init__(self, in_channel: int, out_channel: int, k_neighbours: int) -> None:
        super().__init__()

        self.encoder = Encoder(in_channel=in_channel, k_nearest_neighbours=k_neighbours)
        self.decoder = Decoder(in_channel=out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        pass

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        pass

    def configure_optimizers(self) -> Any:
        pass


class GraphLayer(pl.LightningModule):
    """
    Graph Layer with Local Max Pooling and Feature Map.
    """

    def __init__(self, in_channel: int, out_channel: int, k_neighbours: int = 16) -> None:
        """
        Graph Layer with Local Max Pooling and Feature Map.
        """
        super().__init__()
        self.k: int = k_neighbours
        self.conv: nn.Conv1d = nn.Conv1d(in_channels=in_channel,
                                         out_channels=out_channel,
                                         kernel_size=1)
        self.bn: nn.BatchNorm1d = nn.BatchNorm1d(num_features=out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Graph Layer with Local Max Pooling and Feature Map.

        Args:
            x: (B, N, C) tensor
        Returns:
            x: (B, C, N) tensor
        """
        # Convert input to dense adjacency matrix
        data = Data(x=x.permute(0, 2, 1)) # (B, N, C) -> (B, C, N)

        # Compute k nearest neighbours
        edge_index = knn(x=data.x, k=self.k, batch=data.batch) # (2, E)
        edge_index = edge_index.permute(1, 0) # (E, 2)
        data.edge_index = edge_index

        # Local Max Pooling
        adj_matrix = to_dense_adj(edge_index=data.edge_index, batch=data.batch) # (B, N, N) 
        max_values, _ = torch.max(adj_matrix, dim=2) # (B, N)
        x = torch.bmm(x, max_values.unsqueeze(2)) # (B, N, C) * (B, N, 1) -> (B, N, C)
        x = x.squeeze(2) # (B, N, C) -> (B, N)

        # Feature Map
        x = nn.functional.relu(self.bn(self.conv(x.unsqueeze(2)).squeeze(2))) # (B, N, C) -> (B, C, N) -> (B, C, N) -> (B, C, N)
        return x


class Encoder(pl.LightningModule):
    """
    Encoder of FoldingNet.
    """

    def __init__(self, in_channel: int = 3, k_nearest_neighbours: int = 16) -> None:
        super().__init__()
        self.k = k_nearest_neighbours
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.graph_layer1 = GraphLayer(in_channel=64, out_channel=128, k_neighbours=16)
        self.graph_layer2 = GraphLayer(in_channel=128, out_channel=1024, k_neighbours=16)

        self.conv4 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(num_features=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.size()

        # Compute k nearest neighbours
        data = Data(x=x.permute(0, 2, 1)) # (B, N, C) -> (B, C, N)
        edge_index = knn(x=data.x, k=self.k, batch=data.batch) # (2, E)
        edge_index = edge_index.permute(1, 0) # (E, 2)
        data.edge_index = edge_index

        # Local Max Pooling
        adj_matrix = to_dense_adj(edge_index=data.edge_index, batch=data.batch) # (B, N, N)
        max_values, _ = torch.max(adj_matrix, dim=2) # (B, N)
        x = torch.bmm(x, max_values.unsqueeze(2)) # (B, N, C) * (B, N, 1) -> (B, N, C)
        x = x.squeeze(2) # (B, N, C) -> (B, N)

        # Get covariance matrix, reshape and concatenate with x
        knn_x = index_points(data.x.permute(0, 2, 1), data.edge_index) # (B, N, C) -> (B, E, C)
        mean = torch.mean(knn_x, dim=2, keepdim=True) # (B, E, C) -> (B, E, 1)
        knn_x = knn_x - mean # (B, E, C) - (B, E, 1) -> (B, E, C)
        covariances = torch.matmul(knn_x.transpose(1, 2), knn_x) # (B, C, E) * (B, E, C) -> (B, C, C)
        covariances = covariances.view(b, n, -1).permute(0, 2, 1) # (B, C, C) -> (B, C*C, 1) -> (B, 1, C*C) -> (B, C*C, 1) -> (B, C*C, N)
        x = torch.cat((x, covariances), dim=1) # (B, N) + (B, C*C, N) -> (B, N + C*C, N)

        # three MLP layers
        x = nn.functional.relu(self.bn1(self.conv1(x))) # (B, N + C*C, N) -> (B, 64, N)
        x = nn.functional.relu(self.bn2(self.conv2(x))) # (B, 64, N) -> (B, 64, N)
        x = nn.functional.relu(self.bn3(self.conv3(x))) # (B, 64, N) -> (B, 64, N)

        # two graph layers
        x = self.graph_layer1(x) # (B, 64, N) -> (B, 128, N)
        x = self.graph_layer2(x) # (B, 128, N) -> (B, 1024, N)

        # one MLP layer
        x = nn.functional.relu(self.bn4(self.conv4(x))) # (B, 1024, N) -> (B, 512, N)
        x = torch.max(x, dim=-1)[0]
        return x.squeeze(1)


class FoldingLayer(pl.LightningModule):
    """
    Folding Layer
    """

    def __init__(self, in_channel: int, out_channels: list) -> None:
        super().__init__()

        self.layers = []
        for out_channel in out_channels[:-1]:
            conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
            bn = nn.BatchNorm1d(num_features=out_channel)
            relu = nn.ReLU(inplace=True)
            self.layers.extend([conv, bn, relu])
            in_channel = out_channel
        out_layer = nn.Conv1d(in_channels=in_channel, out_channels=out_channels[-1], kernel_size=1)
        self.layers.append(out_layer)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, grids: torch.Tensor, codewords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grids: (B, 3, N)
            codewords: (B, C, N)
        Returns:
            x: (B, C, N)
        """
        x = torch.cat([grids, codewords], dim=1)
        return self.layers(x)


class Decoder(pl.LightningModule):
    """
    FoldingNet Decoder
    """
    
    def __init__(self, in_channel: int = 512, grid_size: int = 50) -> None:
        super().__init__()

        # Create grid in 2D space
        self.grid = torch.linspace(-0.25, 0.25, grid_size)
        self.grid = torch.meshgrid([self.grid, self.grid]) # (grid_size, grid_size)

        # reshape grid to (2, grid_size*grid_size)

        self.grid = torch.Tensor(self.grid).view(2, -1) # (2, grid_size*grid_size)
        self.grid_size = grid_size

        self.fold1 = FoldingLayer(in_channel=in_channel, out_channels=[512, 512, 3])
        self.fold2 = FoldingLayer(in_channel=in_channel, out_channels=[512, 512, 3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 512)
        Returns:
            grids: (B, 3, grid_size*grid_size)
        """
        batch_size = x.size(0)
        grid = self.grid.unsqueeze(0).repeat(batch_size, 1, 1) # (B, 2, grid_size*grid_size)
        codewords = x.unsqueeze(2).repeat(1, 1, self.grid_size*self.grid_size)

        # folding
        grids = self.fold1(grid, codewords) # (B, 3, grid_size*grid_size)
        grids = self.fold2(grids, codewords) # (B, 3, grid_size*grid_size)
        return grids
