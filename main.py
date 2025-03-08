from models.model.transformer import Transformer
import torch

# tokens ids
tensor = torch.randint(0, 10, (2, 5))

# model
model = Transformer()

src_mask = model.make_padding_mask(tensor, 8)
tgt_mask = model.make_causual_mask(tensor, 8)

out = model(tensor, tensor, src_mask, tgt_mask)

print(out)  # torch.Size([2, 10, 500])