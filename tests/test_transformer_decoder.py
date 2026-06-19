import torch
from model.decoder_layer import DecoderLayer
from model.transformer_decoder import TransformerDecoder

# DecoderLayer 생성
decoder_layer = DecoderLayer(
    d_model=64,
    num_heads=8,
    d_ff=256,
    dropout=0.1
)

# TransformerDecoder 생성 (4개 층)
decoder = TransformerDecoder(decoder_layer, num_layers=4)

# 더미 입력
x = torch.randn(2, 10, 64)

# 실행
output = decoder(x)
print(output.shape)   # torch.Size([2, 10, 64])