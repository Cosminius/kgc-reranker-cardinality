import torch.nn.functional as F


def masked_infonce_loss(logits, labels, false_negative_mask):
    masked = logits.masked_fill(false_negative_mask, float("-inf"))
    return F.cross_entropy(masked, labels)
