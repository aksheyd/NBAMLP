import torch
import torch.nn as nn

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.01
HIDDEN_ACTIVATION = torch.relu
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32


class NBAModel(nn.Module):
    def __init__(
        self, input_size, hidden_size_1=None, hidden_size_2=None, output_size=2
    ):
        super(NBAModel, self).__init__()

        h1 = hidden_size_1 if hidden_size_1 is not None else HIDDEN_SIZE_1
        h2 = hidden_size_2 if hidden_size_2 is not None else HIDDEN_SIZE_2

        self.input_size = input_size
        self.hidden_size_1 = h1
        self.hidden_size_2 = h2
        self.output_size = output_size

        self.hidden_layer_1 = nn.Linear(input_size, h1)
        self.hidden_layer_2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        self.x0 = x

        self.z1 = self.hidden_layer_1(x)
        self.a1 = HIDDEN_ACTIVATION(self.z1)

        self.z2 = self.hidden_layer_2(self.a1)
        self.a2 = HIDDEN_ACTIVATION(self.z2)

        self.z3 = self.output_layer(self.a2)

        return self.z3

    def backwards(self, outputs, labels):
        batch_size = outputs.shape[0]

        one_hot = torch.zeros_like(outputs)
        one_hot[range(batch_size), labels] = 1.0

        dL_dz3 = 2.0 * (outputs - one_hot) / batch_size

        dL_dW3 = dL_dz3.T @ self.a2
        dL_db3 = torch.sum(dL_dz3, dim=0)
        dL_da2 = dL_dz3 @ self.output_layer.weight

        if HIDDEN_ACTIVATION == torch.relu:
            dL_dz2 = dL_da2 * (self.z2 > 0).float()
        elif HIDDEN_ACTIVATION == torch.sigmoid:
            dL_dz2 = dL_da2 * self.a2 * (1.0 - self.a2)
        elif HIDDEN_ACTIVATION == torch.tanh:
            dL_dz2 = dL_da2 * (1.0 - self.a2**2)
        else:
            raise ValueError(f"Unsupported activation: {HIDDEN_ACTIVATION}")

        dL_dW2 = dL_dz2.T @ self.a1
        dL_db2 = torch.sum(dL_dz2, dim=0)
        dL_da1 = dL_dz2 @ self.hidden_layer_2.weight

        if HIDDEN_ACTIVATION == torch.relu:
            dL_dz1 = dL_da1 * (self.z1 > 0).float()
        elif HIDDEN_ACTIVATION == torch.sigmoid:
            dL_dz1 = dL_da1 * self.a1 * (1.0 - self.a1)
        elif HIDDEN_ACTIVATION == torch.tanh:
            dL_dz1 = dL_da1 * (1.0 - self.a1**2)
        else:
            raise ValueError(f"Unsupported activation: {HIDDEN_ACTIVATION}")

        dL_dW1 = dL_dz1.T @ self.x0
        dL_db1 = torch.sum(dL_dz1, dim=0)

        self.output_layer.weight.grad = dL_dW3
        self.output_layer.bias.grad = dL_db3
        self.hidden_layer_2.weight.grad = dL_dW2
        self.hidden_layer_2.bias.grad = dL_db2
        self.hidden_layer_1.weight.grad = dL_dW1
        self.hidden_layer_1.bias.grad = dL_db1
