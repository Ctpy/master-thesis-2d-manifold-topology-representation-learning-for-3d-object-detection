from typing import Any, Optional, List, Tuple, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import index_points, knn
import pytorch_lightning as pl
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points
import open3d

NUM_POINTS = 32

class GraphLayer(pl.LightningModule):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """

    def __init__(self, in_channel, out_channel, k=16):
        super(GraphLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        # KNN
        knn_idx = knn(x, k=self.k)  # (B, N, k)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, k, C)

        # Local Max Pooling
        x = torch.max(knn_x, dim=2)[0]  # (B, N, C)

        # Feature Map
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Encoder(pl.LightningModule):
    """
    Graph based encoder.
    """

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.graph_layer1 = GraphLayer(in_channel=64, out_channel=128, k=16)
        self.graph_layer2 = GraphLayer(in_channel=128, out_channel=1024, k=16)

        self.conv4 = nn.Conv1d(1024, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):
        b, n, c = x.size()

        # get the covariances, reshape and concatenate with x
        knn_idx = knn(x, k=16)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        covariances = (
            torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)
        )
        x = x.permute(0, 2, 1)
        x = torch.cat([x, covariances], dim=1)  # (B, 12, N)

        # three layer MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # two consecutive graph layers
        x = self.graph_layer1(x)
        x = self.graph_layer2(x)

        x = self.bn4(self.conv4(x))

        x = torch.max(x, dim=-1)[0]
        return x


class FoldingLayer(pl.LightningModule):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        grids = grids.to(self.device, dtype=torch.float)
        x = torch.cat([grids, codewords], dim=1)
        # shared mlp
        x = self.layers(x)

        return x


class Decoder(pl.LightningModule):
    """
    Decoder Module of FoldingNet
    """

    def __init__(self, in_channel=512):
        super(Decoder, self).__init__()

        # Sample the grids in 2D space
        xx = np.linspace(-3, 3, NUM_POINTS, dtype=np.float32)
        yy = np.linspace(-3, 3, NUM_POINTS, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)  # (2, 45, 45)

        # reshape
        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 45, 45) -> (2, 45 * 45)

        self.m = self.grid.shape[1]

        self.fold1 = FoldingLayer(in_channel + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(in_channel + 3, [512, 512, 3])

    def forward(self, x):
        """
        x: (B, C)
        """
        batch_size = x.shape[0]

        # repeat grid for batch operation

        grid = self.grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)

        # repeat codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)  # (B, 512, 45 * 45)

        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)  # (B, 3, 45 * 45)

        return recon2


class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 7),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x):
        x = self.encoder(x)
        reconstruction = self.decoder(x)
        # classification = self.classifier(x)
        return reconstruction  # , classification

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        inputs, targets = batch
        if inputs.size(1) > NUM_POINTS**2:
            inputs, _ = sample_farthest_points(inputs, K=NUM_POINTS**2)
        # outputs, classification = self(inputs.to(self.device, dtype=torch.float))
        outputs = self(inputs.to(self.device, dtype=torch.float))
        # class_preds = torch.argmax(classification, dim=1)
        # outputs, _ = sample_farthest_points(outputs.transpose(2, 1), K=400)
        # targets = torch.nn.functional.one_hot(targets, num_classes=7)
        # classification_loss = torch.nn.functional.binary_cross_entropy(classification, targets.float())
        reconstruction_loss, _ = chamfer_distance(outputs.transpose(2, 1), inputs.float())
        # self.log('train_loss', classification_loss + reconstruction_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_classification_loss', classification_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_reconstruction_loss', reconstruction_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # return {'loss': classification_loss + reconstruction_loss, 'classification_loss': classification_loss, 'reconstruction_loss': reconstruction_loss}
        self.log(
            "train_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # self.log('train_classification_loss', classification_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_reconstruction_loss', reconstruction_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": reconstruction_loss}

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        inputs, targets = batch
        if inputs.size(1) > NUM_POINTS**2:
            inputs, _ = sample_farthest_points(inputs, K=NUM_POINTS**2)
        # outputs, classification = self(inputs.to(self.device, dtype=torch.float))
        outputs = self(inputs.to(self.device, dtype=torch.float))
        # outputs, _ = sample_farthest_points(outputs.transpose(2, 1), K=400)
        # targets = torch.nn.functional.one_hot(targets, num_classes=7)
        # class_preds = torch.argmax(classification, dim=1)
        # classification_loss = torch.nn.functional.binary_cross_entropy(
        #     classification, targets.float()
        # )
        reconstruction_loss, _ = chamfer_distance(outputs.transpose(2, 1), inputs.float())
        # self.log(
        #     "val_loss",
        #     classification_loss + reconstruction_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        # self.log(
        #     "val_classification_loss",
        #     classification_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        # self.log(
        #     "val_reconstruction_loss",
        #     reconstruction_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        # return {
        #     "val_loss": classification_loss + reconstruction_loss,
        #     "classification_loss": classification_loss,
        #     "reconstruction_loss": reconstruction_loss,
        # }

        self.log(
            "val_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": reconstruction_loss}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        inputs, targets = batch
        if inputs.size(1) > NUM_POINTS**2:
            inputs, _ = sample_farthest_points(inputs, K=NUM_POINTS**2)
        # outputs, classification = self(inputs.to(self.device, dtype=torch.float))
        outputs = self(inputs.to(self.device, dtype=torch.float))
        # outputs, _ = sample_farthest_points(outputs.transpose(2, 1), K=400)
        # class_preds = torch.argmax(classification, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=7)
        # classification_loss = torch.nn.functional.binary_cross_entropy(
        #    classification, targets.float()
        # )
        reconstruction_loss, _ = chamfer_distance(outputs.transpose(2, 1), inputs.float())
        self.log(
            "test_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_reconstruction_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {
            "test_loss": reconstruction_loss,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer
