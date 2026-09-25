"""Fully connected feed-forward network (adapted from https://github.com/samvanstroud/hepattn)."""

from torch import Tensor, nn


class Dense(nn.Module):
    """``Linear -> SiLU -> ... -> Linear``.

    ``hidden_layers`` is a list of hidden widths; if ``None`` one hidden layer of
    width ``input_size * hidden_dim_scale`` is used.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        hidden_layers: list[int] | None = None,
        hidden_dim_scale: int = 2,
    ) -> None:
        super().__init__()
        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]
        layers: list[nn.Module] = []
        node_list = [input_size, *hidden_layers]
        for i in range(len(node_list) - 1):
            layers.extend((nn.Linear(node_list[i], node_list[i + 1]), nn.SiLU()))
        layers.append(nn.Linear(node_list[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
