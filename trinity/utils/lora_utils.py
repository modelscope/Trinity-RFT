import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
from verl.utils.py_functional import convert_to_regular_types


def key_mapper(key: str) -> str:
    if "base_model.model." in key:
        key = key.replace("base_model.model.", "")
    elif "base_model." in key:
        key = key.replace("base_model.", "")
    return key


def create_dummy_lora(
    model_path: str,
    checkpoint_job_dir: str,
    lora_rank: int,
    lora_alpha: int,
    target_modules: str,
) -> str:
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    lora_config = {
        "task_type": TaskType.CAUSAL_LM,
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": convert_to_regular_types(target_modules),
        "bias": "none",
    }
    peft_model = get_peft_model(model, LoraConfig(**lora_config))
    peft_model.save_pretrained(f"{checkpoint_job_dir}/dummy_lora")
    del model, peft_model
    torch.cuda.empty_cache()
    print(f"!!! [LoRA] Dummy LoRA adapter created at {checkpoint_job_dir}/dummy_lora")
    return f"{checkpoint_job_dir}/dummy_lora"
