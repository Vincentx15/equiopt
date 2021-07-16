import torch
from torch import nn

from equilayers import RegToIrrepConv, IrrepToIrrepConv, RegToRegConv
from equilayers import ToKmerLayer, IrrepActivationLayer, IrrepBatchNorm, RegBatchNorm, IrrepConcatLayer, RegConcatLayer
from equilayers import random_one_hot, reg_action


class RegBinary(nn.Module):
    def __init__(self,
                 pool_size=40,
                 pool_strides=20):
        super(RegBinary, self).__init__()
        model = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=14),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=14),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_strides),
            nn.Flatten(),
            nn.Linear(in_features=752, out_features=1),
            nn.Sigmoid()
        )
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)


class EquiNetBinary(nn.Module):

    def __init__(self,
                 filters=((16, 16), (16, 16), (16, 16)),
                 kernel_sizes=(15, 14, 14),
                 pool_size=40,
                 pool_strides=20,
                 out_size=1,
                 placeholder_bn=False,
                 kmers=1):
        """
        First map the regular representation to irrep setting
        Then goes from one setting to another.
        """
        super(EquiNetBinary, self).__init__()

        # assert len(filters) == len(kernel_sizes)
        # self.input_dense = 1000
        # successive_shrinking = (i - 1 for i in kernel_sizes)
        # self.input_dense = 1000 - sum(successive_shrinking)

        self.kmers = int(kmers)
        self.to_kmer = ToKmerLayer(k=self.kmers)
        reg_in = self.to_kmer.features // 2

        # First mapping goes from the input to an irrep feature space
        first_kernel_size = kernel_sizes[0]
        first_a, first_b = filters[0]
        self.last_a, self.last_b = filters[-1]
        self.reg_irrep = RegToIrrepConv(reg_in=reg_in,
                                        a_out=first_a,
                                        b_out=first_b,
                                        kernel_size=first_kernel_size)
        self.first_bn = IrrepBatchNorm(a=first_a, b=first_b, placeholder=placeholder_bn)
        self.first_act = IrrepActivationLayer(a=first_a, b=first_b)

        # Now add the intermediate layer : sequence of conv, BN, activation
        self.irrep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        for i in range(1, len(filters)):
            prev_a, prev_b = filters[i - 1]
            next_a, next_b = filters[i]
            self.irrep_layers.append(IrrepToIrrepConv(
                a_in=prev_a,
                b_in=prev_b,
                a_out=next_a,
                b_out=next_b,
                kernel_size=kernel_sizes[i],
            ))
            self.bn_layers.append(IrrepBatchNorm(a=next_a, b=next_b, placeholder=placeholder_bn))
            # Don't add activation if it's the last layer
            # placeholder = (i == len(filters) - 1)
            # placeholder = True
            self.activation_layers.append(IrrepActivationLayer(a=next_a,
                                                               b=next_b))

        self.concat = IrrepConcatLayer(a=self.last_a, b=self.last_b)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_strides)
        self.flattener = nn.Flatten()
        self.dense = nn.Linear(in_features=1504, out_features=out_size)

    def forward(self, inputs):

        x = self.to_kmer(inputs)
        x = self.reg_irrep(x)
        x = self.first_bn(x)
        x = self.first_act(x)

        # rcinputs = reg_action(inputs)
        # rcx = self.reg_irrep(rcinputs)
        # rcx = self.first_bn(rcx)
        # rcx = self.first_act(rcx)

        for irrep_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
            x = irrep_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

            # rcx = irrep_layer(rcx)
            # rcx = bn_layer(rcx)
            # rcx = activation_layer(rcx)

        # Average two strands predictions
        x = self.concat(x)
        x = self.pool(x)
        x = self.flattener(x)
        outputs = self.dense(x)

        # rcx = self.concat(rcx)
        # rcx = self.pool(rcx)
        # rcx = self.flattener(rcx)
        # rcout = self.dense(rcx)
        # print(outputs)
        # print(rcout)
        return outputs


class CustomRCPS(nn.Module):

    def __init__(self,
                 filters=(16, 16, 16),
                 kernel_sizes=(15, 14, 14),
                 pool_size=40,
                 pool_strides=20,
                 out_size=1,
                 placeholder_bn=False,
                 kmers=1):
        """
        First map the regular representation to irrep setting
        Then goes from one setting to another.
        """
        super(CustomRCPS, self).__init__()

        self.kmers = int(kmers)
        self.to_kmer = ToKmerLayer(k=self.kmers)
        reg_in = self.to_kmer.features // 2
        filters = [reg_in] + list(filters)

        # Now add the intermediate layer : sequence of conv, BN, activation
        self.reg_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        for i in range(len(filters) - 1):
            prev_reg = filters[i]
            next_reg = filters[i + 1]
            self.reg_layers.append(RegToRegConv(
                reg_in=prev_reg,
                reg_out=next_reg,
                kernel_size=kernel_sizes[i],
            ))
            self.bn_layers.append(RegBatchNorm(reg_dim=next_reg, placeholder=placeholder_bn))
            # Don't add activation if it's the last layer
            placeholder = (i == len(filters) - 1)
            self.activation_layers.append(nn.ReLU())

        self.concat = RegConcatLayer(reg=filters[-1])
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_strides)
        self.flattener = nn.Flatten()
        self.dense = nn.Linear(in_features=752, out_features=out_size)

    def forward(self, inputs):
        x = self.to_kmer(inputs)
        for reg_layer, bn_layer, activation_layer in zip(self.reg_layers, self.bn_layers, self.activation_layers):
            x = reg_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

        # Average two strands predictions, pool and go through Dense
        x = self.concat(x)
        x = self.pool(x)
        x = self.flattener(x)
        x = self.dense(x)
        outputs = torch.sigmoid(x)
        return outputs


class PosthocModel(nn.Module):
    def __init__(self, model):
        super(PosthocModel, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        rcout = self.model(reg_action(x))
        return (out + rcout) / 2


class AverageModel(nn.Module):
    def __init__(self, model1, model2):
        super(PosthocModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        out = self.model(x)
        out2 = self.model2(x)
        return (out + out2) / 2


def post_hoc_from_model(model):
    return PosthocModel(model=model)


if __name__ == '__main__':
    pass
    x = random_one_hot(size=(2, 1000))
    rcps = CustomRCPS()
    y = rcps(x)

    equinet = EquiNetBinary()
    y = equinet(x)

    regnet = RegBinary()
    y = regnet(x)

    posthoc = post_hoc_from_model(regnet)
    y = posthoc(x)
