import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Enable access to hugging face
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

from LoRA_config import lora_config as LoRA_Config
lora_config = LoRA_Config()

from transformers import AutoTokenizer, LlamaForCausalLM, set_seed
model = LlamaForCausalLM.from_pretrained(lora_config.model_name, dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(lora_config.model_name) 
tokenizer.pad_token = tokenizer.eos_token

device = next(model.parameters()).device
model_input = tokenizer(lora_config.eval_promt, return_tensors="pt").to(device)
model.eval()

# Set seed for reproducibility
set_seed(lora_config.seed)

# Perform beam-search
import torch
with torch.inference_mode():
    print(tokenizer.decode(
        model.generate(
            **model_input, 
            max_new_tokens=lora_config.max_new_tokens, 
            num_beams=lora_config.num_beams, 
            do_sample=False
        )[0], 
        skip_special_tokens=True
    ))