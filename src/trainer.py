import torch.nn.functional as F
from transformers import Trainer

from .loss import masked_infonce_loss


class UnmaskedKGCRerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        n, k, L = inputs["input_ids"].shape
        flat = {key: v.view(-1, L) for key, v in inputs.items()}
        outputs = model(**flat)
        logits = outputs.logits.view(n, k)
        loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


class MaskedKGCRerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        mask = inputs.pop("false_negative_mask")
        labels = inputs.pop("labels")
        n, k, L = inputs["input_ids"].shape
        flat = {key: v.view(-1, L) for key, v in inputs.items()}
        outputs = model(**flat)
        logits = outputs.logits.view(n, k)
        loss = masked_infonce_loss(logits, labels, mask)
        return (loss, outputs) if return_outputs else loss
