import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-fr'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)
attention_mask = inputs['attention_mask']

torch.onnx.export(
    model,                               
    (inputs['input_ids'], attention_mask, decoder_input_ids),
    "model.onnx",               
    input_names=["input_ids", "attention_mask"],  
    output_names=["logits"],             
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "attention_mask": {0: "batch_size", 1: "sequence_length"}, "logits": {0: "batch_size", 1: "sequence_length"}}, 
    opset_version=11                
)