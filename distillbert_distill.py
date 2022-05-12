from operator import mod
from statistics import mode
import torch
import transformers
from copy import deepcopy
from torch import device
from typing import Optional, List, Tuple
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import Transformer, Embeddings
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions

class Seequential(torch.nn.Sequential):
    def forward(self, **kwargs):
        input = self[0](**kwargs)
        for module in self[1:]:
            input = module(input)
        return input

def ditilbert_to_sequential(model, inputs):
    layers = []
    # for child in model.children():
    #     if isinstance(child, transformers.DistilBertModel):
    #         inner_layers = _ditilbert_to_sequential(child)
    #         layers.extend(inner_layers)
    #     else:
    #         layers.append(child)
    model_children = list(model.children())

    layers.append(_ditilbert_to_sequential(model_children[1]))
    layers.append(model_children[0])
    layers.extend(model_children[2:])

    sequential_layers = Seequential(*[deepcopy(d) for d in layers])
    #print(sequential_layers)
    model.eval()
    sequential_layers.eval()
    with torch.no_grad():
        orig_output = model(**inputs)
        #print(orig_output)
        seq_output = sequential_layers(**inputs)
        #print(seq_output)
    assert torch.allclose(orig_output.logits, seq_output.logits)
    return sequential_layers

def _ditilbert_to_sequential(module):
    layers = []
    for child in module.children():
        if isinstance(child, Embeddings):
            layers.append(DistilBertModelInit(child))
        elif isinstance(child, Transformer):
            layers.extend(list(child.layer))
        else:
            layers.append(child)
    return layers


class TransformerFirstBlock(torch.nn.Module):
    def __init__(self, embedding, config):

class DistilBertModelInit(torch.nn.Module):
    def __init__(self, embedding, config):
        super().__init__()
        self.embeddings = embedding
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
        
        transformer_input = [inputs_embeds, attention_mask, head_mask, 
                                output_attentions, output_hidden_states, return_dict]
        
        return transformer_input

def main():
    #configuration = BertConfig()
    student_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    print(student_model)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    print(inputs)
    sequential_layers = ditilbert_to_sequential(student_model)
    
if __name__ == '__main__':
    main()