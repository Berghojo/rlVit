
import torch
import torch.nn as nn
import numpy as np

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)
class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape

        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq.to(tensor.device))
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.value_factor = 0.5
        self.entropy_factor = 0.01
    def forward(self, policy_per_action, values, rewards, policy):
        target_value = self.get_values(rewards, values)
        #advantage = target_value - values.squeeze()

        clipped_policy = torch.clip(policy, 1e-5, 1 - 1e-5)
        clipped_policy_per_action = torch.clip(policy_per_action, 1e-5, 1 - 1e-5)

        #value_loss = torch.mean(advantage ** 2)
        policy_loss = -torch.mean(torch.log(clipped_policy_per_action) * target_value)

        entropy = -(torch.sum(policy * torch.log(clipped_policy), dim=1))

        entropy_loss = -torch.mean(entropy)
        loss = policy_loss + self.entropy_factor * entropy_loss
        return loss, policy_loss, entropy_loss

    def get_values(self, reward, values):
        gamma = 0.9
        pos_reward = 1
        neg_reward = -1
        n_step_return = 197
        max_size = values.shape[1]
        r = reward.expand(-1, max_size)
        r2 = r.clone()
        r2[r >= 0] = pos_reward
        r2[r < 0] = neg_reward
        # max_size = values.shape[1]
        # seq_len = r.shape[1]
        # for i in range(1, seq_len):
        #     for e in range(i + 1, i + 1 + n_step_return):
        #         if e < max_size:

        #             if (e - i) == n_step_return:
        #                 val[:, i] += (gamma ** (e - i)) * values[:, e]
        #             else:
        #                 val[:, i] += (gamma ** (e - i)) * r[:, e]

        return r2