import torch.nn as nn
import torch
from typing import Tuple


class MMMidSlowFastHead(nn.Module):
    def __init__(
        self,
        dim_in: Tuple[int, int],
        num_classes: Tuple[int, int],
        act_func: str = "softmax",
    ) -> None:
        super(MMMidSlowFastHead, self).__init__()
        self.num_classes = num_classes
        self.dim_in = dim_in

        F = 4608
        V, N = num_classes

        self.projection_verb = nn.Linear(F, V, bias=True)
        self.projection_noun = nn.Linear(F, N, bias=True)

        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{act_func} is not supported as an activation function.")

    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate the features
        x = torch.cat((audio_features, video_features), dim=-1)

        x_v = self.projection_verb(x)
        x_n = self.projection_noun(x)

        x_v = self.fc_inference(x_v, self.act)
        x_n = self.fc_inference(x_n, self.act)

        return (x_v, x_n)

    def fc_inference(self, x: torch.Tensor, act: nn.Module) -> torch.Tensor:
        """
        Perform fully convolutional inference.

        Args:
            x (tensor): input tensor.
            act (nn.Module): activation function.

        Returns:
            tensor: output tensor.
        """
        if not self.training:
            x = act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)

        return x
