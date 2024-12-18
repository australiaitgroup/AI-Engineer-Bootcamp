import gradio as gr
import openai
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def ask(text,history):
  history = history or []
  tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
  model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1",torch_dtype=torch.bfloat16)

  prompt = f"<human>: {text}\n<bot>:"
  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

  input_length = inputs.input_ids.shape[1]
  outputs = model.generate(**inputs, max_new_tokens=48, temperature=0.7,
                           return_dict_in_generate=True)

  tokens = outputs.sequences[0, input_length:]
  output = tokenizer.decode(tokens)
  return output

with gr.Blocks() as server:
  with gr.Tab("LLM Inferencing"):

    model_input = gr.Textbox(label="Your Question:",
                             value="What’s your question?", interactive=True)
    ask_button = gr.Button("Ask")
    model_output = gr.Textbox(label="The Answer:", interactive=False,
                              value="Answer goes here...")

  ask_button.click(ask, inputs=[model_input], outputs=[model_output])

server.launch()