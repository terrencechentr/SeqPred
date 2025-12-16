from dataclasses import dataclass
from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence

@dataclass
class DataCollatorForSept:
    """
    Data collator for Sept model.
    """
    label_pad: float = float("nan")

    def __call__(self, features: List[Dict[str, List[float]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.float) for f in features]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.label_pad)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        } 