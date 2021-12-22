from torch import nn
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=(9,)),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=(9,)),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 128, kernel_size=(9,)),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        # Dummy input the size of the actual input to collect the size of embeds in the size_forward path
        dummy_input = torch.rand((1, 3, 1400))
        self.embed_size = self.size_forward(dummy_input)

        self.FC = nn.Sequential(
            nn.Linear(self.embed_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    # Method that is required by the DA package to collect the embeds from the network
    def get_embed_size(self):
        return [self.embed_size]

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.FC(features)
        # in this example features are the embeds
        # if embeds of more than one layer is required, then features should be a list
        # the shape of embeds should be the same as the features above.
        return features, logits

    # if you know the size of embed this forward is not necessary
    def size_forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        return features.shape[1]
