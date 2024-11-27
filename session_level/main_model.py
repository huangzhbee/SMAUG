import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_model


class soft_attn(nn.Module):
    def __init__(self, dim):
        super(soft_attn, self).__init__()

        self.weight1 = nn.Linear(dim, dim, bias=False)
        self.weight2 = nn.Linear(dim, dim, bias=True)
        self.weight_c = nn.Linear(dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, L, emb_dim = x.shape
        mean_embed = torch.mean(x, dim=1).unsqueeze(1).expand(-1, L, -1)  # (B,L,dim)
        attn = self.weight1(mean_embed) + self.weight2(x)
        attn = self.sigmoid(attn)
        attn = self.weight_c(attn)  # (B,L,1)

        out = torch.matmul(attn.permute(0, 2, 1), x).squeeze(1)

        return out


class model_base(nn.Module):
    def __init__(self, config, device, loc_emb_dic=None):
        super().__init__()
        self.device = device

        # session embedding
        self.emb_session_dim = 32
        # parameters for condition
        self.emb_time_dim = config["model"]["timeemb"]
        self.loc_emb_all = loc_emb_dic
        self.emb_loc_dim = 32

        # ST embedding
        self.cond_total_dim = self.emb_time_dim + self.emb_loc_dim
        # user embedding
        self.emb_user_dim = self.cond_total_dim

        # module for user embedding
        self.User_embedding = soft_attn(self.emb_user_dim)

        # parameters for diffusion models
        config_diff = config["diffusion"]
        config_diff["cond_dim"] = self.cond_total_dim

        input_dim = 1
        self.diffmodel = diff_model(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha = 1 - self.beta
        self.alpha_hat = np.cumprod(self.alpha)
        self.alpha_torch = torch.tensor(self.alpha_hat).float().to(self.device).unsqueeze(1)

        # loss function
        self.denoise_loss = nn.MSELoss()
        self.emb_loss = nn.MSELoss(reduce=False)

    def time_embedding(self, pos, d_model=128):
        if pos.dim() == 1:
            pe = torch.zeros(pos.shape[0], d_model).to(self.device)
            position = pos.unsqueeze(1)
            div_term = 1 / torch.pow(
                10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
            position = pos.unsqueeze(2)
            div_term = 1 / torch.pow(
                10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
            )
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)
            pe[pos == 48] = 0

        return pe

    def loc_embedding(self, loc_ids):
        if loc_ids.dim() == 1:
            B = loc_ids.shape[0]
            loc_emb = torch.cat([
                    torch.index_select(self.loc_emb_all, dim=0, index=(loc_ids[b] - 1)) for b in range(B)],
                dim=0)
        else:
            B, L = loc_ids.shape
            loc_emb = torch.cat([
                torch.cat([
                    torch.index_select(self.loc_emb_all, dim=0, index=loc_ids[b, l] - 1) for l in range(L)],
                    dim=0).unsqueeze(0) for b in range(B)],
                dim=0)
        return loc_emb

    def get_ST_info(self, tp, emb, loc):
        B, emb_dim = emb.shape

        time_embed = self.time_embedding(tp, self.emb_time_dim)  # (B,time_emb_dim)
        loc_embed = self.loc_embedding(loc)  # (B,loc_emb_dim)

        time_embed = time_embed.unsqueeze(1).expand(-1, emb_dim, -1)  # (B,emb_dim,tim_emb_dim)
        loc_embed = loc_embed.unsqueeze(1).expand(-1, emb_dim, -1)  # (B,emb_dim,loc_emb_dim)

        side_info = torch.cat([time_embed, loc_embed], dim=-1)  # (B,emb_dim,cond_total_dim)

        side_info = side_info.permute(0, 2, 1)  # (B,*,emb_dim)

        return side_info

    def get_user_info(self, emb, st_traj):
        B, emb_dim = emb.shape

        s = st_traj[:, :, 0]
        t = st_traj[:, :, 1]
        time_embed = self.time_embedding(t)  # (B,L,time_emb_dim)
        loc_embed = self.loc_embedding(s)  # (B,L,loc_emb_dim)
        user_emb = torch.cat((loc_embed, time_embed), dim=-1)
        user_emb = self.User_embedding(user_emb)
        user_emb = user_emb.unsqueeze(1).expand(-1, emb_dim, -1).permute(0, 2, 1)  # (B,*,emb_dim)
        return user_emb
    def set_input_to_diffmodel(self, noisy_data):
        total_input = noisy_data.unsqueeze(1)  # (B,1,*,*)
        return total_input

    def calc_loss_valid(self, tp, loc, emb, ST_traj, is_train):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate cate_loss for all t
            loss = self.calc_loss(tp, loc, emb, ST_traj, is_train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, tp, loc, emb, ST_traj, is_train, set_t=-1):

        B, emb_dim = emb.shape
        loss = 0.0

        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(emb)
        noisy_data = (current_alpha ** 0.5) * emb + (1.0 - current_alpha) ** 0.5 * noise

        # current ST
        ST_info = self.get_ST_info(tp, emb, loc)

        # user embedding
        user_info = self.get_user_info(emb, ST_traj)

        total_input = self.set_input_to_diffmodel(noisy_data)  # (B,1,emb)
        predict_noise = self.diffmodel(total_input, user_info, ST_info, t)  # (B,emb)

        predict_emb = (noisy_data - (1.0 - current_alpha) ** 0.5 * predict_noise) / (current_alpha ** 0.5)

        loss_1 = self.denoise_loss(noise, predict_noise)
        loss_2 = self.emb_loss(predict_emb, emb).mean(axis=1)

        alpha = 0.1
        lambda_t = 1 - alpha * t / self.num_steps
        loss += loss_1 + (lambda_t * loss_2).mean().item()

        return loss

    def impute(self, tp, loc, emb, ST_traj, n_samples):

        B, emb_dim = emb.shape
        emb_samples = torch.zeros(B, n_samples, emb_dim).to(self.device)

        for i in range(n_samples):

            current_sample = torch.randn_like(emb)

            # current ST
            ST_info = self.get_ST_info(tp, emb, loc)

            # user embedding
            user_info = self.get_user_info(ST_traj)

            for t in range(self.num_steps - 1, -1, -1):
                diff_input = current_sample
                diff_input = diff_input.unsqueeze(1)  # (B,1,emb_dim)

                predicted = self.diffmodel(diff_input, user_info, ST_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            emb_samples[:, i] = current_sample.detach()

        return emb_samples

    def forward(self, batch, is_train=1):
        (
            tp,
            loc,
            emb,
            ST_traj,
            session_id
        ) = self.process_data(batch)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(tp, loc, emb, ST_traj, is_train)

    def evaluate(self, batch, n_samples):
        (
            tp,
            loc,
            emb,
            ST_traj,
            session_id
        ) = self.process_data(batch)

        with torch.no_grad():

            emb_sample = self.impute(tp, loc, emb, ST_traj, n_samples)

        return emb_sample, emb, session_id


class Model_Shanghai(model_base):
    def __init__(self, config, device, loc_emb_dict):
        super(Model_Shanghai, self).__init__(config, device, loc_emb_dict)

    def process_data(self, batch):
        tp = batch["timepoints"].to(self.device)  # (B,1)
        loc = batch["locations"].to(self.device)  # (B,1)
        emb = batch["session_emb"].to(self.device).float()  # (B,emb_dim)
        ST_traj = batch["ST_traj"].to(self.device).permute(0, 2, 1)  # (B,150,2)
        session_id = batch["session_id"].to(self.device)  # (B,1)

        return (
            tp,
            loc,
            emb,
            ST_traj,
            session_id
        )