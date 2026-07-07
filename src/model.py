"""
Model definition — base model + LoRA adapter configuration.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model and LoRA training."""

    # Base model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 512

    # QLoRA settings
    use_4bit: bool = True               # 4-bit quantization
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"    # NormalFloat4 — best for LLMs

    # LoRA settings
    lora_r: int = 16                     # Rank — higher = more capacity, more VRAM
    lora_alpha: int = 32                 # Scaling factor (usually 2x rank)
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 4 * 4 = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3

    # Output
    output_dir: str = "models/medical-specialist"
    logging_steps: int = 25


def get_quantization_config(config: ModelConfig):
    """
    Create BitsAndBytes quantization config for 4-bit loading.

    NOTE: bitsandbytes requires a CUDA GPU. This is for Colab training.
    For CPU inference, load the model without quantization.
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization — saves more memory
    )


def load_base_model(config: ModelConfig, quantize: bool = True):
    """
    Load the base model with optional quantization.

    For training (Colab with GPU): quantize=True
    For inference (CPU): quantize=False
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {}
    if quantize:
        model_kwargs["quantization_config"] = get_quantization_config(config)
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    if quantize:
        model.config.use_cache = False  # Required for gradient checkpointing

    return model, tokenizer


def setup_lora(model, config: ModelConfig):
    """
    Apply LoRA adapters to the model.

    This wraps specific layers (attention projections) with
    small trainable matrices. The base model stays frozen.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Prepare model for quantized training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model, lora_config


def load_fine_tuned_model(
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_path: str = "models/medical-specialist",
):
    """
    Load base model + fine-tuned LoRA adapter for inference.

    The adapter is tiny (~10MB) — the base model is large (~2GB).
    In production, you'd cache the base model and swap adapters.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model (no quantization for CPU inference)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # Merge adapter into base for faster inference

    return model, tokenizer


def count_parameters(model) -> dict:
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": round(100 * trainable / total, 2),
    }
