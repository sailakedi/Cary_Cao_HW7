"""
academic_qa_multiturn.py

End-to-end pipeline:

1) --phase generate
   - Sample 100 arXiv abstracts from gfissore/arxiv-abstracts-2021 (diverse categories). :contentReference[oaicite:2]{index=2}
   - For each paper, call GPT-4.x to create 5 multi-turn academic Q&A dialogs
     (with at least one hallucination / edge-case question per paper).
   - Save dataset as instruction-tuning JSONL: each line = 1 full dialog
     formatted as:
       <|system|>...<|user|>Q1<|assistant|>A1<|user|>Q2<|assistant|>A2...

2) --phase finetune
   - Fine-tune unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit (Instruct, 4-bit) :contentReference[oaicite:3]{index=3}
     with QLoRA via Unsloth + TRL SFTTrainer on "synthetic_qa_multiturn.jsonl".

3) --phase evaluate
   - Compare base vs fine-tuned model on a small set of test academic questions.
"""

import os
import json
import time
import random
import argparse
from typing import List, Dict, Any

import torch
from openai import OpenAI
from datasets import load_dataset

# from unsloth import FastLanguageModel, is_bfloat16_supported
SE_UNSLOTH = False
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    USE_UNSLOTH = True
except NotImplementedError:
    # On Mac / non-supported GPU, Unsloth raises here
    USE_UNSLOTH = False
except Exception:
    # Any other import errors
    USE_UNSLOTH = False

from trl import SFTTrainer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# ---------------- CONFIG ---------------- #

OPENAI_MODEL_NAME = "gpt-4.1-mini"  # or "gpt-4.1" if you want max quality

# ArXiv abstracts dataset (HF)
ARXIV_DATASET_NAME = "gfissore/arxiv-abstracts-2021"
NUM_PAPERS = 100

# Multi-turn dialog settings
DIALOGS_PER_PAPER = 5
MIN_TURNS_PER_DIALOG = 2  # user→assistant turns (so 4 actual turns)
MAX_TURNS_PER_DIALOG = 3

SYNTHETIC_QA_JSONL = "synthetic_qa_multiturn.jsonl"

TRAIN_SYSTEM_PROMPT = (
    "You are a helpful academic Q&A assistant specialized in scholarly content. "
    "You answer clearly and precisely based only on the given paper abstract."
)

# Use an Instruct 4-bit model (good for chat-style tasks). :contentReference[oaicite:4]{index=4}
BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
FINETUNED_MODEL_DIR = "llama3.1-8b-academic-qa-qlora"

MAX_SEQ_LENGTH = 2048
LR = 2e-4
NUM_EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACC_STEPS = 4

# ---------------- OPENAI + DATA ---------------- #

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    return OpenAI(api_key=api_key)


def stratified_sample_arxiv(num_papers: int = NUM_PAPERS, seed: int = 42):
    """
    Sample ~NUM_PAPERS abstracts with a rough spread over high-level arXiv domains.
    The dataset has a 'categories' field like 'cs.LG math.ST', etc. :contentReference[oaicite:5]{index=5}
    """
    print(f"Loading dataset '{ARXIV_DATASET_NAME}' ...")
    ds = load_dataset(ARXIV_DATASET_NAME, split="train")
    print(f"Dataset size: {len(ds)}")

    random.seed(seed)
    domains = ["cs", "math", "physics", "stat", "econ", "q-bio", "q-fin", "eess"]
    per_domain_target = max(1, num_papers // len(domains))

    # Group indices by domain
    by_domain = {d: [] for d in domains}
    misc = []
    for i, row in enumerate(ds):
        cats = row.get("categories", [])
        if isinstance(cats, str):
            cats = cats.split()
        found = False
        for c in cats:
            prefix = c.split(".")[0]
            if prefix in by_domain:
                by_domain[prefix].append(i)
                found = True
                break
        if not found:
            misc.append(i)

    chosen_indices = []
    for d in domains:
        idxs = by_domain[d]
        random.shuffle(idxs)
        chosen_indices.extend(idxs[:per_domain_target])

    # If not enough, fill from misc
    if len(chosen_indices) < num_papers:
        remaining = num_papers - len(chosen_indices)
        random.shuffle(misc)
        chosen_indices.extend(misc[:remaining])
    else:
        chosen_indices = chosen_indices[:num_papers]

    sampled = ds.select(sorted(chosen_indices))
    print(f"Stratified sample size: {len(sampled)}")
    return sampled


def build_dialog_generation_prompt(
    title: str,
    abstract: str,
    categories: List[str],
    dialogs_per_paper: int = DIALOGS_PER_PAPER,
) -> str:
    """
    Prompt GPT-4.x to generate multi-turn dialogs:
    {
      "dialogs": [
        {
          "turns": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
          ]
        },
        ...
      ]
    }
    """
    categories_str = ", ".join(categories) if categories else "Unspecified"
    prompt = f"""
You are a research assistant that reads academic papers and creates quiz-style Q&A dialogs.

You are given the TITLE, ARXIV CATEGORIES, and ABSTRACT of a research paper.
Using ONLY the information in the abstract, generate EXACTLY {dialogs_per_paper} multi-turn dialogs
about this paper in the following JSON format:

{{
  "dialogs": [
    {{
      "turns": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}},
        ...
      ]
    }},
    ...
  ]
}}

Requirements:

1. Number of dialogs & length:
   - Produce exactly {dialogs_per_paper} dialogs.
   - Each dialog should have between {MIN_TURNS_PER_DIALOG} and {MAX_TURNS_PER_DIALOG} question–answer pairs
     (which means 2× that many turns, alternating "user" and "assistant").
   - The first turn in each dialog MUST be from "user", and roles must alternate strictly:
       user -> assistant -> user -> assistant -> ...

2. Content style:
   - The user asks about the paper; the assistant explains based only on the abstract.
   - Questions should be academic / exam-style:
       * factual (What, When, Who, Where),
       * conceptual (Why, How, What is the intuition),
       * methodological (How is the model evaluated, What architecture is used),
       * interpretive (What do the results suggest, What are the limitations).
   - Across all dialogs, cover key points: motivation, main contributions, methods, results, implications,
     and (if mentioned) limitations or future work.
   - Some questions should explicitly reference imagined sections or figures, e.g.:
       "According to Section 3 of the paper, what is the main contribution of the proposed method?"
       "Based on Figure 2 (as described in the abstract), how does performance compare to baselines?"

3. Hallucination / edge cases:
   - Across the entire output (all {dialogs_per_paper} dialogs), include at least ONE question that contains
     an incorrect or hallucinated detail OR asks for numeric information not present in the abstract.
   - In the corresponding assistant answer, you MUST:
       * Correct the false premise and/or
       * Explicitly state that the abstract does not provide the requested detail.
     Example:
       user: "According to the abstract, what is the exact value of constant XYZ used in the loss function?"
       assistant: "The abstract does not specify any constant named XYZ or its value; this detail is not provided."

4. Faithfulness:
   - All non-edge-case answers must be supported directly by the abstract.
   - Do NOT invent results, datasets, numerical scores, or methods not clearly implied by the abstract.
   - Use clear, concise academic language.

5. Output formatting:
   - Output *only* one valid JSON object following the schema above.
   - Use double quotes for all JSON strings.
   - Do NOT add any text or explanation outside of the JSON.

TITLE: {title}
CATEGORIES: {categories_str}

ABSTRACT:
\"\"\"{abstract}\"\"\"
"""
    return prompt.strip()


def call_openai_for_dialogs(client: OpenAI, prompt: str) -> List[Dict[str, Any]]:
    """
    Call GPT-4.x, expecting JSON: {"dialogs": [ {"turns": [...]}, ... ]}
    """
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You produce strict JSON as instructed."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            dialogs = data.get("dialogs", [])
            cleaned_dialogs = []

            for d in dialogs:
                turns = d.get("turns", [])
                cleaned_turns = []
                last_role = None
                for t in turns:
                    role = t.get("role", "").strip()
                    text = t.get("content", "").strip()
                    if role not in ["user", "assistant"] or not text:
                        continue
                    # Enforce alternation
                    if last_role and role == last_role:
                        continue
                    cleaned_turns.append({"role": role, "content": text})
                    last_role = role

                # Ensure at least one Q-A
                if len(cleaned_turns) >= 2 and cleaned_turns[0]["role"] == "user":
                    cleaned_dialogs.append({"turns": cleaned_turns})

            if not cleaned_dialogs:
                raise ValueError("No valid dialogs found in JSON.")
            return cleaned_dialogs

        except Exception as e:
            print(f"[WARN] Dialog generation failed on attempt {attempt+1}: {e}")
            time.sleep(3)

    raise RuntimeError("Failed to generate dialogs after 3 attempts.")


def dialogs_to_training_lines(dialogs: List[Dict[str, Any]]) -> List[str]:
    """
    Convert a list of dialogs into training text lines.
    Format:
      <|system|>SYSTEM_PROMPT<|user|>Q1<|assistant|>A1<|user|>Q2<|assistant|>A2...
    """
    lines = []
    for dialog in dialogs:
        turns = dialog.get("turns", [])
        if not turns:
            continue
        s = f"<|system|>{TRAIN_SYSTEM_PROMPT}"
        for t in turns:
            if t["role"] == "user":
                s += f"<|user|>{t['content']}"
            else:
                s += f"<|assistant|>{t['content']}"
        lines.append(s)
    return lines


def generate_synthetic_dataset():
    client = get_openai_client()
    sampled = stratified_sample_arxiv(NUM_PAPERS)

    num_papers = len(sampled)
    total_dialogs = 0
    total_lines = 0

    with open(SYNTHETIC_QA_JSONL, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(sampled):
            title = row.get("title", "").strip()
            abstract = row.get("abstract", "").strip()
            cats = row.get("categories", [])
            if isinstance(cats, str):
                cats = cats.split()

            if not abstract:
                print(f"[SKIP] Paper {i} has empty abstract")
                continue

            print(f"\n=== [{i+1}/{num_papers}] Generating dialogs for paper ===")
            print(f"Title: {title[:120]}...")
            prompt = build_dialog_generation_prompt(title, abstract, cats)
            dialogs = call_openai_for_dialogs(client, prompt)

            print(f"  -> got {len(dialogs)} dialogs")
            total_dialogs += len(dialogs)

            lines = dialogs_to_training_lines(dialogs)
            for line in lines:
                out_f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
            total_lines += len(lines)

            # light rate-limit
            time.sleep(1.0)

    print(f"\nDone. Papers processed: {num_papers}")
    print(f"Total dialogs: {total_dialogs}")
    print(f"Total JSONL entries (one per dialog): {total_lines}")
    print(f"Saved to: {SYNTHETIC_QA_JSONL}")


# ---------------- FINE-TUNING (QLoRA) ---------------- #

def load_text_dataset_for_sft(jsonl_path: str, val_split: float = 0.1):
    """
    Load JSONL as HF dataset and split into train / eval.
    val_split is the fraction held out for validation.
    """
    full = load_dataset("json", data_files=jsonl_path, split="train")
    full = full.shuffle(seed=3407)

    n = len(full)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    train_ds = full.select(range(n_train))
    eval_ds  = full.select(range(n_train, n))

    print(f"Loaded {n} examples from {jsonl_path}")
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    return train_ds, eval_ds


def finetune_llama_with_qlora(val_split: float = 0.1):
    print("Loading base model (Instruct, 4-bit) via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("Attaching LoRA adapters (QLoRA)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    train_ds, eval_ds = load_text_dataset_for_sft(SYNTHETIC_QA_JSONL, val_split=val_split)

    training_args = TrainingArguments(
        output_dir=FINETUNED_MODEL_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        warmup_steps=5,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",     # <-- NEW: run eval each epoch
        eval_steps=None,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to=[],                    # no W&B unless you want it
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,            # <-- NEW
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=training_args,
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training finished.")
    print(trainer_stats)

    print("Running final evaluation on dev set...")
    eval_metrics = trainer.evaluate()
    print("Eval metrics:", eval_metrics)

    print(f"Saving fine-tuned adapters + tokenizer to {FINETUNED_MODEL_DIR} ...")
    model.save_pretrained(FINETUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINETUNED_MODEL_DIR)
    print("Done.")

# ---------------- EVALUATION ---------------- #

DEFAULT_TEST_QUESTIONS = [
    "What is the main motivation behind the method proposed in this paper?",
    "How do the authors evaluate their proposed approach, and what metrics do they use?",
    "According to the experiments, how does the method compare to baseline techniques?",
    "What assumptions or limitations do the authors mention about their approach?",
    "Explain one core concept or definition from the paper in simple terms.",
    "How does the paper position itself with respect to prior work in the field?",
    "What kind of dataset or data source do the authors use, and why?",
    "What are the key contributions claimed by the authors?",
    "How does the paper's method generalize to other tasks or domains, if at all?",
    "If you had to critique this paper, what potential weaknesses would you highlight?",
]


def load_models_for_eval():
    print("Loading base model for evaluation...")
    if USE_UNSLOTH:
        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(base_model)

        print("Loading fine-tuned model for evaluation...")
        ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
            model_name=FINETUNED_MODEL_DIR,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
       )
    
        FastLanguageModel.for_inference(ft_model)

        return base_model, base_tokenizer, ft_model, ft_tokenizer
    else:
        # Fallback: use OpenAI GPT-4o or your API of choice
        from openai import OpenAI
        client = OpenAI()  # assumes OPENAI_API_KEY in env
        return client, None, "openai"


def generate_answer(model, tokenizer, question: str, device: str = "cuda") -> str:
    prompt = f"<|system|>{TRAIN_SYSTEM_PROMPT}<|user|>{question}<|assistant|>"
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    if "<|assistant|>" in decoded:
        return decoded.split("<|assistant|>")[-1].strip()
    return decoded.replace(prompt, "").strip()


def evaluate_models(test_questions: List[str] = None):
    if test_questions is None:
        test_questions = DEFAULT_TEST_QUESTIONS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_model, base_tok, ft_model, ft_tok = load_models_for_eval()
    base_model.to(device)
    ft_model.to(device)

    for q in test_questions:
        print("=" * 80)
        print(f"Q: {q}\n")

        base_ans = generate_answer(base_model, base_tok, q, device=device)
        ft_ans = generate_answer(ft_model, ft_tok, q, device=device)

        print("Base Model Answer:\n-------------------")
        print(base_ans)
        print("\nFine-tuned Model Answer:\n------------------------")
        print(ft_ans)
        print("\n")


# ---------------- CLI ---------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn academic Q&A synthetic dataset + QLoRA fine-tuning"
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "finetune", "evaluate", "all"],
        default="all",
    )
    parser.add_argument(
        "--test_questions_file",
        type=str,
        default=None,
        help="Optional file with one test question per line for evaluation.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation fraction for fine-tuning (e.g., 0.1 = 10%).",
    )
    args = parser.parse_args()

    if args.phase in ("generate", "all"):
        print("\n=== PHASE 1: Generate multi-turn synthetic Q&A dataset ===")
        generate_synthetic_dataset()

    if args.phase in ("finetune", "all"):
        print("\n=== PHASE 2: Fine-tune Llama 3.1 8B Instruct with QLoRA ===")
        finetune_llama_with_qlora(val_split=args.val_split)

    if args.phase in ("evaluate", "all"):
        print("\n=== PHASE 3: Evaluate base vs fine-tuned model ===")
        if args.test_questions_file and os.path.exists(args.test_questions_file):
            with open(args.test_questions_file, "r", encoding="utf-8") as f:
                test_qs = [line.strip() for line in f if line.strip()]
        else:
            test_qs = None
        evaluate_models(test_questions=test_qs)

if __name__ == "__main__":
    main()
