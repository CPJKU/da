from torch import nn
import math


def calc_out_size(in_size, padding, kernel_size, dilation, stride):
    return math.floor((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class Net(nn.Module):
    def __init__(self, inp_size=28, n_outs=10):
        super().__init__()
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 48, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        ])

        c1_size = calc_out_size(inp_size, 0, 5, 1, 1)
        m1_size = calc_out_size(c1_size, 0, 2, 1, 2)
        c2_size = calc_out_size(m1_size, 0, 5, 1, 1)
        m2_size = calc_out_size(c2_size, 0, 2, 1, 2)

        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(m2_size * m2_size * 48, 100),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
            )
        ])

        self.out_layer = nn.Linear(100, n_outs)

        self.embeds_size = [m1_size*m1_size*32, m2_size*m2_size*48, 100, 100]

    def get_embed_size(self):
        return self.embeds_size

    def forward(self, x):
        embeds = []
        for layer in self.feature_extractor:
            x = layer(x)
            embeds.append(x.view(x.size(0), -1))
        x = x.view(x.size(0), -1)
        for lin_layer in self.classifier:
            x = lin_layer(x)
            embeds.append(x)
        out = self.out_layer(x)
        return embeds, out
