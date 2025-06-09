
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

st.title("🔥 GPT-2 Text Generator (Open Source)")

prompt = st.text_input("Enter your prompt:", "Once upon a time...")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=100, do_sample=True)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

st.subheader("✨ Generated Text:")
st.write(generated_text)
