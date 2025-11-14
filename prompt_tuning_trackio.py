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

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name) 
tokenizer.pad_token = tokenizer.eos_token

# Get Dataset
from samsum_dataset import get_preprocessed_samsum_for_prompt_tuning
train_dataset = get_preprocessed_samsum_for_prompt_tuning({}, tokenizer, "train[:100]")
test_dataset = get_preprocessed_samsum_for_prompt_tuning({}, tokenizer, "validation[:100]")

eval_prompt = """
    Dialog:
    A: Hi Tom, are you busy tomorrow’s afternoon?
    B: I’m pretty sure I am. What’s up?
    A: Can you go with me to the animal shelter?.
    B: What do you want to do?
    A: I want to get a puppy for my son.
    B: That will make him so happy.
    A: Yeah, we’ve discussed it many times. I think he’s ready now.
    B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) 
    A: I'll get him one of those little dogs.
    B: One that won't grow up too big;-)
    A: And eat too much;-))
    B: Do you know which one he would like?
    A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
    B: I bet you had to drag him away.
    A: He wanted to take it home right away ;-).
    B: I wonder what he'll name it.
    A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
    ---
    Summary:
    """
model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)

model = LlamaForCausalLM.from_pretrained(
    train_config.model_name, 
    dtype="auto", 
    device_map="auto", 
    use_cache=True if torch.cuda.is_available() else False
)

for num_workers in [0,2]:
    for pin_memory in [True, False]:
        for fp16 in [True, False]:

            print(f"\n\n---- With num_workers={num_workers}, pin_memory={pin_memory}, fp16={fp16} ----\n\n")

            # Configure LoRA
            from peft import TaskType, get_peft_model, PromptEmbedding, PromptTuningConfig, PromptTuningInit
            prompt_tuning_init_text = "Summarize this dialogue"
            config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                prompt_tuning_init_text=prompt_tuning_init_text,
                num_virtual_tokens=6,
                tokenizer_name_or_path=train_config.model_name
            )

            model = get_peft_model(model, config)
            model.print_trainable_parameters()

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
                output_dir="prompt_tuning1",
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                per_device_train_batch_size=8 if torch.cuda.is_available() else 1, # On MPS devices, I need to set the batch size to 1 to avoid memory issues.
                per_device_eval_batch_size=8 if torch.cuda.is_available() else 1, # On MPS devices, I need to set the batch size to 1 to avoid memory issues.
                seed=train_config.seed,
                
                fp16=fp16,#True if torch.cuda.is_available() else False,
                dataloader_pin_memory=pin_memory,#True if torch.cuda.is_available() else False,
                dataloader_num_workers=num_workers,#4 if torch.cuda.is_available() else 0,

                report_to="trackio",
                project="prompt-tuning-optimization",
                trackio_space_id="emoudahi/my_space",
                run_name=f"prompt-tuning-fp16={fp16}-pin_memory={pin_memory}-num_workers={num_workers}"
            )

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,          # dynamic padding to longest in batch
                pad_to_multiple_of=8 if torch.cuda.is_available() else None
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

            # # Set seed for reproducibility
            # set_seed(train_config.seed)

            # # Perform beam-search
            # model.eval()
            # with torch.inference_mode():
            #     print(tokenizer.decode(
            #         model.generate(
            #             **model_input, 
            #             max_new_tokens=train_config.max_new_tokens, 
            #             num_beams=train_config.num_beams, 
            #             do_sample=False
            #         )[0], 
            #         skip_special_tokens=True
            #     ))

            # model.unload()

            model = model.base_model