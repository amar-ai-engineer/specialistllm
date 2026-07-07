# SpecialistLLM - Medical Q&A Fine-Tuning with QLoRA

**Fine-tune a 1.1B parameter LLM into a medical specialist by training only 1.2% of its parameters.**

Generic LLM gives you "diabetes makes you tired." The fine-tuned version gives you polyuria, polydipsia, HbA1c screening criteria, and risk factors — with proper structure.

## What It Does

Takes TinyLlama (a general-purpose 1.1B LLM) and fine-tunes it on 5,000 medical Q&A pairs from NIH medical websites using QLoRA. The result: a model that answers medical questions with proper terminology, structured responses, and clinical accuracy.

The adapter weights are only ~10MB. The base model stays frozen. Training takes ~25 minutes on a free Google Colab T4 GPU.

## Why QLoRA?

Full fine-tuning of even a small 1.1B model needs ~4.4GB VRAM and updates all 1.1 billion parameters. That's wasteful — you're not changing everything the model knows, just adding medical knowledge.

QLoRA solves this:
1. **Quantization** — load the model in 4-bit (75% memory savings)
2. **LoRA** — add small trainable matrices to attention layers (1.2% of params)
3. **Result** — same quality, fraction of the cost

| Approach | Parameters Trained | Memory Needed | Training Time |
|----------|-------------------|---------------|---------------|
| Full fine-tuning | 1.1B (100%) | ~4.4 GB | Hours |
| **QLoRA** | **~13M (1.2%)** | **~0.55 GB** | **~25 min** |

## Results

The fine-tuned model produces responses that are:
- **3-4x longer** with more detail
- **Structured** with headings, bullet points, and tables
- **Medically accurate** with proper terminology
- **Clinically useful** with diagnostic criteria and treatment options

See the side-by-side comparison in the Streamlit app.

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Base Model | TinyLlama 1.1B | Small enough for free Colab T4 |
| Fine-Tuning | QLoRA (PEFT + bitsandbytes) | Train 1.2% of params |
| Dataset | MedQuad (NIH medical Q&A) | 47K real medical Q&A pairs |
| Training | Google Colab (free T4) | Free GPU |
| Demo UI | Streamlit | Quick to build |

## How to Run

### Streamlit Demo (no GPU needed)
```bash
git clone https://github.com/amar-ai-engineer/specialistllm.git
cd specialistllm
pip install -r requirements.txt
streamlit run app.py
```

The demo app shows pre-computed comparisons and supports live inference via Gemini API.

### Training (Google Colab)
1. Open `notebooks/training.ipynb` in Google Colab
2. Set runtime to T4 GPU (free tier)
3. Run all cells (~25 minutes)
4. Download the adapter weights (~10MB)
5. Place in `models/medical-specialist/`

## Project Structure

```
specialistllm/
├── app.py                    # Streamlit UI — model comparison demo
├── notebooks/
│   └── training.ipynb        # Colab notebook for QLoRA fine-tuning
├── src/
│   ├── dataset.py            # Dataset loading and formatting
│   ├── model.py              # Model + LoRA configuration
│   ├── evaluate.py           # Comparison and quality metrics
│   └── inference.py          # Inference pipeline (local + API fallback)
├── data/
│   └── comparison_results.json  # Pre-computed demo results
├── models/                   # Fine-tuned adapter weights (after training)
├── requirements.txt
└── .env.example
```

## Key Design Decisions

- **TinyLlama over larger models**: Fits on free Colab T4. The concepts (QLoRA, LoRA, PEFT) are identical for 7B or 70B models — just change the model name.
- **QLoRA over full LoRA**: 4-bit quantization means the base model uses ~75% less memory. No quality loss for this task.
- **Rank 16**: Good balance between capacity and efficiency. Rank 8 works for simple tasks, rank 32+ for complex ones.
- **Attention layers only**: q_proj, k_proj, v_proj, o_proj. These are where the model learns relationships between tokens — the most impactful layers to adapt.
- **3 epochs**: Enough to learn the medical Q&A pattern without overfitting to the 5K training examples.

## When to Fine-Tune vs When Not To

| Situation | Best Approach |
|-----------|--------------|
| Want formatted output | Prompt engineering |
| Need domain knowledge | RAG (retrieval) |
| Need domain reasoning | Fine-tuning (LoRA) |
| Need fundamental behavior change | Full fine-tuning |

Fine-tuning is for when the model needs to **think** like a specialist, not just **reference** documents.

## Built By

**Amar Ismail** — AI Engineer
- [LinkedIn](https://www.linkedin.com/in/amar-ai-engineer/)
- [GitHub](https://github.com/amar-ai-engineer)
- amarismail522@gmail.com
