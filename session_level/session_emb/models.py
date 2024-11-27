import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from model_init import Model_init
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

urbanclip_init = Model_init(
    dim=512,  # model dimension
    num_tokens=20000,  # number of text tokens
    unimodal_depth=6,  # depth of the unimodal transformer
    # depth of the multimodal transformer
    dim_head=64,  # dimension per attention head
    heads=8,  # number of attention heads
).cuda()

# test
text = torch.randint(0, 20000, (4, 512)).cuda()
emb = torch.randn(4, 3, 256, 256).cuda()

loss = urbanclip_init(
    text=text,
    images=emb,
    return_loss=True
)
loss.backward()

logits = urbanclip_init(
    text=text,
    images=emb
)

text_embeds, session_embeds = urbanclip_init(
    text=text,
    images=emb,
    return_embeddings=True
)
