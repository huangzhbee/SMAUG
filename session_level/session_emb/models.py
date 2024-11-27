import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from model_init import Model_init
from transformers import BertTokenizer

# 加载预训练的BERT分词器
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
images = torch.randn(4, 3, 256, 256).cuda()

loss = urbanclip_init(
    text=text,
    images=images,
    return_loss=True  # set this to True to get the full caption + contrastive loss
)
loss.backward()

logits = urbanclip_init(
    text=text,
    images=images
)

text_embeds, image_embeds = urbanclip_init(
    text=text,
    images=images,
    return_embeddings=True
)
