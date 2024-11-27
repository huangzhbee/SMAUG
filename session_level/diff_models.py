import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_model(nn.Module):
    def __init__(self, config, inputdim=1):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    cond_dim=config["cond_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"]
                )
                for _ in range(config["layers"])
            ]
        )

        self.skip_projection = nn.Conv1d(self.channels, self.channels, 1)
        self.output_projection = nn.Conv1d(self.channels, 1, 1)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, user_info, ST_info, diffusion_step):
        B, inputdim, emb_dim = x.shape

        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, user_info, ST_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)  # (B,channel,emb_dim)
        x = F.relu(x)
        x = self.output_projection(x)  # (B,1,emb_dim)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, cond_dim, channels, diffusion_embedding_dim):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        self.ST_cond_projection = Conv1d_with_init(cond_dim, 2 * channels, 1)
        self.user_cond_projection = Conv1d_with_init(cond_dim, 2 * channels, 1)

        self.transAm = get_torch_trans(layers=1, channels=channels)

        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

    def forward(self, x, user_info, ST_info, diffusion_emb):
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.transAm(y)

        y = self.mid_projection(y)  # (B,2*channel,L * emb_dim)

        ST_info = self.ST_cond_projection(ST_info)
        y = y + ST_info

        user_info = self.user_cond_projection(user_info)
        y = y + user_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,L * emb_dim)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / math.sqrt(2.0), skip
