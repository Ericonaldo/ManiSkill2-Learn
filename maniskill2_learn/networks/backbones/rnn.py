import torch.nn as nn

from maniskill2_learn.utils.meta import get_logger
from maniskill2_learn.utils.torch import ExtendedModule, load_checkpoint

from ..builder import BACKBONES
from ..modules import build_activation_layer


@BACKBONES.register_module()
class SimpleRNN(ExtendedModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout=0.2,
        hidden_dim=256,
        n_layers=2,
        rnn_type="LSTM",
        act_cfg=dict(type="ReLU"),
        pretrained=None,
    ):
        super(SimpleRNN, self).__init__()

        self.rnn = getattr(nn, rnn_type)(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout
        )
        self.act = build_activation_layer(act_cfg)
        self.last_fc = nn.Linear(hidden_dim, output_dim)
        self.init_weights(pretrained)

    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        out = self.last_fc(self.act(out[:, -1]))
        return out, h

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
