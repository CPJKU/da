import torch.nn as nn
from collections import OrderedDict


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softmax":
        return nn.Softmax()
    elif activation == "log_softmax":
        return nn.LogSoftmax(dim=1)
    else:
        return None


def single_unit(in_dim, out_dim, activation, dropout_rate):
    """Creates a linear layer with the respective dimensions, activation function and a dropout rate."""
    linear = nn.Linear(in_dim, out_dim)
    unit = [("linear", linear)]
    if activation is not None:
        unit.append(("activation", activation))
    if dropout_rate > 0.0:
        unit.append(("dropout", nn.Dropout(p=dropout_rate)))
    return nn.Sequential(OrderedDict(unit))


def build_units(layers, activation, dropout_rate):
    """Creates a fully connected network based on layers, activation functions and dropout rate."""
    units = []
    for ind in range(len(layers)):
        # last layers built separately
        if ind + 1 < len(layers):
            units.append(single_unit(layers[ind], layers[ind + 1], activation, dropout_rate))
    return nn.Sequential(*units) if len(units) > 0 else nn.Sequential()
