import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Enable access to
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

from LoRA_config import lora_config as TRAIN_CONFIG
train_config = TRAIN_CONFIG()

from transformers import AutoTokenizer, LlamaForCausalLM, set_seed, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
model = LlamaForCausalLM.from_pretrained(
    train_config.model_name, 
    dtype="auto", 
    device_map="auto", 
    use_cache=True if torch.cuda.is_available() else False,
)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.mps.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name) 
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
from peft import LoHaConfig, TaskType, get_peft_model
config = LoHaConfig(

)
model = get_peft_model(model, config)
# print(model)
model.print_trainable_parameters()

# Get Dataset
from samsum_dataset import get_preprocessed_samsum
train_dataset = get_preprocessed_samsum({}, tokenizer, "train[1:10]")
test_dataset = get_preprocessed_samsum({}, tokenizer, "validation[1:10]")

import torch.optim as optim
optimizer = optim.AdamW(
            model.parameters(),
)

from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(
    optimizer, 
    step_size=1, 
    gamma=0.85
)

training_args = TrainingArguments(
    output_dir="LoHa_experiment1",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    per_device_train_batch_size=8 if torch.cuda.is_available() else 1, # On MPS devices, I need to set the batch size to 1 to avoid memory issues.
    per_device_eval_batch_size=8 if torch.cuda.is_available() else 1, # On MPS devices, I need to set the batch size to 1 to avoid memory issues.
    seed=train_config.seed,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,          # dynamic padding to longest in batch
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)

trainer = Trainer(
    model=model,
    optimizers=(optimizer, scheduler),
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
)
trainer.train()

# import torch
model.save_pretrained("LoHa_experiment1")
model = LlamaForCausalLM.from_pretrained(
            "LoHa_experiment1",
            device_map=device,
            use_cache=True if torch.cuda.is_available() else True,
            dtype="auto",
            is_decoder=True
        )

device = next(model.parameters()).device
model_input = tokenizer(train_config.eval_promt, return_tensors="pt").to(device)

# Set seed for reproducibility
set_seed(train_config.seed)

# Perform beam-search
model.eval()
with torch.inference_mode():
    print(tokenizer.decode(
        model.generate(
            **model_input, 
            max_new_tokens=train_config.max_new_tokens, 
            num_beams=train_config.num_beams, 
            do_sample=False
        )[0], 
        skip_special_tokens=True
    ))