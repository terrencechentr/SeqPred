from transformers import PreTrainedTokenizer
from typing import List, Dict, Union
import torch


class NumericTokenizer(PreTrainedTokenizer):
    def __init__(self, min_val=0.0, max_val=10.0, num_boxes=100, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.num_boxes = num_boxes
        self.pad_token = '0'

    def _tokenize(self, values: List[str]) -> List[int]:
        tokens = []
        for value in values:
            tokens.append(self._convert_token_to_id(value))
        return tokens

    def _convert_token_to_id(self, token):
        value = float(token)
        clamped_value = max(self.min_val, min(self.max_val, value))
        normalized = (clamped_value - self.min_val) / (self.max_val - self.min_val)
        
        return int(normalized * (self.num_boxes - 1))
        

    def _convert_id_to_token(self, index):
        value = self.min_val + index * (self.max_val - self.min_val) / (self.num_boxes - 1)
        return str(value)
    

    def convert_tokens_to_string(self, tokens):
        return ','.join([str(t) for t in tokens])

    def __call__(self, data: list[str], padding=False, truncation=False, max_length=None, return_tensors=None, **kwargs) -> Dict[str, Union[List[int], torch.Tensor]]:

        # Process each sequence in the batch
        all_tokens = []
        all_attention_masks = []

        data = [sequence.split(',') for sequence in data]
        
        # Find the maximum length in the data if padding is True
        if padding and max_length:
            max_length = min(max(len(sequence) for sequence in data), max_length)
            assert max_length == SEQUENCE_LENGTH
        
        for sequence in data:
            tokens = self._tokenize(sequence)
            if truncation and max_length:
                tokens = tokens[:max_length]
            
            attention_mask = [1] * len(tokens)
            
            if padding and max_length:
                pad_len = max_length - len(tokens)
                tokens += [0] * pad_len
                attention_mask += [0] * pad_len
            
            all_tokens.append(tokens)
            all_attention_masks.append(attention_mask)
        
        result = {
            "input_ids": all_tokens,
            "attention_mask": all_attention_masks
        }

        return result


    def get_vocab(self):
        return {
            "vocab": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
            }
        }

    def get_vocab_size(self):
        return self.num_boxes

