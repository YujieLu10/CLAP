from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import torch.nn as nn
from icecream import ic

from soft_embedding import SoftEmbedding
import os
n_tokens = 20
initialize_from_vocab = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

s_wte = SoftEmbedding(model.get_input_embeddings(), 
                      n_tokens=n_tokens, 
                      initialize_from_vocab=initialize_from_vocab)

model.set_input_embeddings(s_wte)

model.load_state_dict(torch.load(os.path.join("checkpoints/causal/020141", "model_19.pth")))
inputs = tokenizer("", return_tensors="pt")

inputs['input_ids'] = torch.cat([torch.full((1,n_tokens), 50256), inputs['input_ids']], 1).to(device)
inputs['attention_mask'] = torch.cat([torch.full((1,n_tokens), 1), inputs['attention_mask']], 1).to(device)
inputs = inputs.to(device)

sampling_params = \
        {
            "max_tokens": 100,
            "temperature": 0.1,
            "top_p": 0.9,
            "num_return_sequences": 10,
            "repetition_penalty": 1.2,
            'use_cache': False,
            'output_scores': True,
            'return_dict_in_generate': True,
            'do_sample': True,
        }
prompt_len = inputs['input_ids'].shape[-1]
output_dict = model.generate(inputs['input_ids'], max_length=prompt_len + sampling_params['max_tokens'], **sampling_params)
