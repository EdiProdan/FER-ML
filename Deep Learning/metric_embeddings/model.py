import torch.nn as nn
from torch.nn import TripletMarginLoss


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(in_channels=num_maps_in, out_channels=num_maps_out, kernel_size=k, padding='same',
                              bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.model = nn.Sequential(
            _BNReluConv(num_maps_in=input_channels, num_maps_out=emb_size, k=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(num_maps_in=emb_size, num_maps_out=emb_size, k=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(num_maps_in=emb_size, num_maps_out=emb_size, k=3),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

    def get_features(self, img):
        """Returns tensor with dimensions BATCH_SIZE, EMB_SIZE"""
        return self.model(img).squeeze()

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        return TripletMarginLoss()(a_x, p_x, n_x)
