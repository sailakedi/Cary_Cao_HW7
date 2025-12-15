"""
merge_and_export.py

Merges the QLoRA adapters into the base Llama 3.1 8B Instruct model and
saves a full Hugging Face model that can be converted to GGUF or used by Ollama.
"""

import torch
# from unsloth import FastLanguageModel
SE_UNSLOTH = False
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except NotImplementedError:
    # On Mac / non-supported GPU, Unsloth raises here
    USE_UNSLOTH = False
except Exception:
    # Any other import errors
    USE_UNSLOTH = False

BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"  # FP16 / BF16 base
ADAPTER_DIR = "llama3.1-8b-academic-qa-qlora"            # your fine-tuned adapters
MERGED_OUT_DIR = "llama3.1-8b-academic-qa-merged"

def main():
    # Load the base full-precision model (not 4-bit)
    print("Loading base full-precision model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=2048,
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        load_in_4bit=False,
    )

    # Load LoRA adapter weights
    print("Loading LoRA adapters from:", ADAPTER_DIR)
    model.load_adapter(ADAPTER_DIR)

    # Merge LoRA into base weights
    print("Merging LoRA adapters into base model weights...")
    FastLanguageModel.merge_and_unload(model)

    print("Saving merged model to:", MERGED_OUT_DIR)
    model.save_pretrained(MERGED_OUT_DIR)
    tokenizer.save_pretrained(MERGED_OUT_DIR)
    print("Done. You can now convert MERGED_OUT_DIR to GGUF or use with Ollama.")

if __name__ == "__main__":
    main()
