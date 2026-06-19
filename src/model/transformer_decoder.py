import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        # num_layers만큼 DecoderLayer를 복제해서 쌓음
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x, mask=None):
        # 각 DecoderLayer를 순차적으로 통과
        for layer in self.layers:
            x = layer(x, mask)
        return x