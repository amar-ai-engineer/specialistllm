"""
SpecialistLLM — Medical Q&A Fine-Tuning Demo
Streamlit app comparing base vs fine-tuned model responses.
"""

import streamlit as st
import json
import os

st.set_page_config(
    page_title="SpecialistLLM - Medical Q&A",
    page_icon="🧬",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');
    
    .stApp { 
        background-color: #ffffff; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1 { 
        font-weight: 800 !important; 
        color: #0f172a !important; 
        letter-spacing: -0.025em;
    }
    h2, h3 { 
        color: #115e59 !important; 
        font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0fdfa;
        border-right: 1px solid #ccfbf1;
    }

    /* Model Cards */
    .stInfo {
        background-color: #f8fafc !important;
        color: #334155 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    .stSuccess {
        background-color: #f0fdfa !important;
        color: #134e4a !important;
        border: 1px solid #ccfbf1 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        border-left: 5px solid #0d9488 !important;
    }

    /* Technical Metrics */
    [data-testid="stMetricValue"] {
        color: #0d9488 !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Code Blocks */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }

    /* Input & Buttons */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        background-color: #0d9488 !important;
        color: white !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #0f766e !important;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.2) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load Demo Data ───────────────────────────────────────
@st.cache_data
def load_demo_results():
    """Load pre-computed comparison results."""
    demo_path = "data/comparison_results.json"
    if os.path.exists(demo_path):
        with open(demo_path) as f:
            return json.load(f)

    # Fall back to built-in demo data
    from src.evaluate import create_demo_results
    return create_demo_results()


# ── Page Selection ───────────────────────────────────────
page = st.radio(
    "Navigate",
    ["Compare Models", "Try It Live", "How QLoRA Works"],
    horizontal=True,
)

# ══════════════════════════════════════════════════════════
# PAGE 1: Compare Models
# ══════════════════════════════════════════════════════════
if page == "Compare Models":
    st.title("Base Model vs Fine-Tuned Model")
    st.markdown("Same question, very different answers. See how fine-tuning transforms a generic LLM into a medical specialist.")

    results = load_demo_results()

    for i, r in enumerate(results):
        st.markdown(f"### Q: {r['question']}")
        st.caption(f"Topic: {r['topic']}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Base Model** (generic)")
            st.info(r["base_response"])
            bq = r.get("base_quality", {})
            st.caption(
                f"Words: {bq.get('word_count', '?')} | "
                f"Medical terms: {bq.get('medical_terms_found', '?')} | "
                f"Score: {bq.get('quality_score', '?')}/10"
            )

        with col2:
            st.markdown("**Fine-Tuned Model** (medical specialist)")
            st.success(r["finetuned_response"])
            fq = r.get("finetuned_quality", {})
            st.caption(
                f"Words: {fq.get('word_count', '?')} | "
                f"Medical terms: {fq.get('medical_terms_found', '?')} | "
                f"Score: {fq.get('quality_score', '?')}/10"
            )

        st.markdown("---")

    # Summary metrics
    st.markdown("### Overall Comparison")
    results_with_quality = [r for r in results if "base_quality" in r and "finetuned_quality" in r]
    if results_with_quality:
        avg_base_score = sum(r["base_quality"]["quality_score"] for r in results_with_quality) / len(results_with_quality)
        avg_ft_score = sum(r["finetuned_quality"]["quality_score"] for r in results_with_quality) / len(results_with_quality)
        avg_base_words = sum(r["base_quality"]["word_count"] for r in results_with_quality) / len(results_with_quality)
        avg_ft_words = sum(r["finetuned_quality"]["word_count"] for r in results_with_quality) / len(results_with_quality)

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Base Avg Score", f"{avg_base_score:.1f}/10")
        with metric_cols[1]:
            st.metric("Fine-Tuned Avg Score", f"{avg_ft_score:.1f}/10")
        with metric_cols[2]:
            st.metric("Base Avg Words", f"{avg_base_words:.0f}")
        with metric_cols[3]:
            st.metric("Fine-Tuned Avg Words", f"{avg_ft_words:.0f}")

# ══════════════════════════════════════════════════════════
# PAGE 2: Try It Live
# ══════════════════════════════════════════════════════════
elif page == "Try It Live":
    st.title("Ask a Medical Question")
    st.markdown("Compare a generic response vs a specialist response in real-time using Gemini API.")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("Set GOOGLE_API_KEY to try live inference.")
        user_key = st.text_input("Or paste your Gemini API key:", type="password")
        if user_key:
            os.environ["GOOGLE_API_KEY"] = user_key
            st.rerun()

    # Sample questions
    st.markdown("**Quick examples:**")
    sample_cols = st.columns(3)
    sample_q = ""
    with sample_cols[0]:
        if st.button("Diabetes symptoms"):
            sample_q = "What are the common symptoms of Type 2 diabetes?"
    with sample_cols[1]:
        if st.button("Stroke warning signs"):
            sample_q = "What are the warning signs of a stroke?"
    with sample_cols[2]:
        if st.button("Anemia treatment"):
            sample_q = "What causes iron deficiency anemia and how is it treated?"

    question = st.text_area(
        "Your medical question:",
        value=sample_q,
        placeholder="e.g., What are the treatment options for hypertension?",
    )

    if st.button("Compare Responses", type="primary") and question and os.getenv("GOOGLE_API_KEY"):
        from src.inference import MedicalSpecialist, get_base_model_response

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Generic Model Response**")
            with st.spinner("Generating generic response..."):
                base_result = get_base_model_response(question)
            st.info(base_result["answer"])
            st.caption(f"Model: {base_result['model']} | Time: {base_result['time_seconds']}s")

        with col2:
            st.markdown("**Medical Specialist Response**")
            with st.spinner("Generating specialist response..."):
                specialist = MedicalSpecialist(use_local=False)
                specialist_result = specialist.ask(question)
            st.success(specialist_result["answer"])
            st.caption(f"Model: {specialist_result['model']} | Time: {specialist_result['time_seconds']}s")

    st.markdown("---")
    st.caption(
        "Note: Live demo uses Gemini API with different system prompts to simulate the "
        "base vs fine-tuned difference. The actual fine-tuned model runs from the Colab notebook."
    )

# ══════════════════════════════════════════════════════════
# PAGE 3: How QLoRA Works
# ══════════════════════════════════════════════════════════
elif page == "How QLoRA Works":
    st.title("How QLoRA Fine-Tuning Works")

    st.markdown("""
    ### The Problem
    Fine-tuning a large language model normally requires updating **all** its parameters.
    For a 7B model, that means ~28GB of GPU memory — an expensive A100 GPU.

    ### The Solution: LoRA + Quantization

    **Step 1: Quantization (the Q in QLoRA)**

    Load the model in 4-bit precision instead of 16-bit.
    This reduces memory by ~75%.

    | Precision | Memory for 1.1B model |
    |-----------|----------------------|
    | FP32 (full) | ~4.4 GB |
    | FP16 (half) | ~2.2 GB |
    | INT8 | ~1.1 GB |
    | **INT4 (QLoRA)** | **~0.55 GB** |

    **Step 2: LoRA (Low-Rank Adaptation)**

    Instead of updating the full weight matrix W, add two small matrices A and B:
    """)

    st.code("""
    # Original weight matrix: W (4096 x 4096) = 16.7M parameters
    # LoRA decomposition:
    #   A (4096 x 16) = 65K parameters
    #   B (16 x 4096) = 65K parameters
    #   Total: 130K parameters (99.2% reduction!)
    #
    # Output = W @ x + (A @ B) @ x
    # The base model (W) stays frozen. Only A and B are trained.
    """, language="python")

    st.markdown("""
    ### Why This Works

    The key insight: **weight updates during fine-tuning have low rank**.

    When you fine-tune a model on medical Q&A, you're not changing everything the model
    knows. You're adding a thin layer of medical knowledge on top. That thin layer can be
    captured by small matrices (rank 16 is enough for most tasks).

    ### Our Training Setup

    | Setting | Value | Why |
    |---------|-------|-----|
    | Base model | TinyLlama 1.1B | Small enough for free Colab T4 |
    | Quantization | 4-bit NF4 | NormalFloat4 is optimal for LLM weights |
    | LoRA rank | 16 | Good balance of capacity vs efficiency |
    | LoRA alpha | 32 | 2x rank is standard |
    | Target modules | q, k, v, o projections | Attention layers matter most |
    | Learning rate | 2e-4 | Standard for LoRA |
    | Epochs | 3 | Enough to learn patterns, not memorize |
    | Dataset | 5,000 medical Q&A | From NIH medical websites |

    ### The Result

    - **Trainable**: ~13M params (1.2% of total)
    - **Adapter size**: ~10 MB (vs ~2 GB full model)
    - **Training time**: ~25 minutes on free Colab T4
    - **Quality**: Significantly better medical answers
    """)

    st.markdown("""
    ### When to Use Fine-Tuning vs Prompting

    | Approach | Best For | Cost |
    |----------|----------|------|
    | **Prompt engineering** | Formatting, style, simple instructions | Free |
    | **RAG** (retrieval) | Adding knowledge without changing the model | Medium |
    | **Fine-tuning (LoRA)** | Teaching new skills, domain specialization | Low (with QLoRA) |
    | **Full fine-tuning** | Fundamental behavior changes | High |

    For most use cases, start with prompting. If that's not enough, try RAG.
    Fine-tuning is for when you need the model to deeply understand a domain —
    not just reference documents, but think like a specialist.
    """)

# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Amar Ismail | SpecialistLLM — QLoRA Fine-Tuning for Medical Q&A")
