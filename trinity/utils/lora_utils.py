def create_dummy_lora(
    model_path: str,
    checkpoint_job_dir: str,
    lora_rank: int,
    lora_alpha: int,
    target_modules: str,
) -> str:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    lora_config = {
        "task_type": TaskType.CAUSAL_LM,
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "bias": "none",
    }
    peft_model = get_peft_model(model, LoraConfig(**lora_config))
    peft_model.save_pretrained(f"{checkpoint_job_dir}/dummy_lora")
    del model, peft_model
    torch.cuda.empty_cache()
    return f"{checkpoint_job_dir}/dummy_lora"
