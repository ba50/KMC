import torch
from torch import nn


class SimpleFC(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleFC, self).__init__()
        self.fc_1_x_branch = nn.Linear(input_size, 256)
        self.fc_2_x_branch = nn.Linear(256, 128)
        self.fc_3_x_branch = nn.Linear(128, 64)
        self.fc_4_x_branch = nn.Linear(64, 32)
        self.fc_out_x_branch = nn.Linear(32, output_size)

        self.fc_1_y_branch = nn.Linear(input_size, 256)
        self.fc_2_y_branch = nn.Linear(256, 128)
        self.fc_3_y_branch = nn.Linear(128, 64)
        self.fc_4_y_branch = nn.Linear(64, 32)
        self.fc_out_y_branch = nn.Linear(32, output_size)

        self.activation = nn.Tanh()

        self.drop_out = nn.Dropout(0.05)

    def forward(self, x_in, y_in):
        x = self.fc_1_x_branch(x_in)
        x = self.activation(x)

        x = self.fc_2_x_branch(x)
        x = self.activation(x)

        x = self.fc_3_x_branch(x)
        x = self.activation(x)

        x = self.fc_4_x_branch(x)
        x = self.activation(x)

        x = self.drop_out(x)

        x = self.fc_out_x_branch(x)
        x = self.activation(x)

        y = self.fc_1_y_branch(y_in)
        y = self.activation(y)

        y = self.fc_2_y_branch(y)
        y = self.activation(y)

        y = self.fc_3_y_branch(y)
        y = self.activation(y)

        y = self.fc_4_y_branch(y)
        y = self.activation(y)

        y = self.drop_out(y)

        y = self.fc_out_y_branch(y)
        y = self.activation(y)

        out = x + y
        return out
