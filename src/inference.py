"""
Inference pipeline — run the fine-tuned model on new questions.
"""

import os


class MedicalSpecialist:
    """
    Inference wrapper for the fine-tuned medical specialist model.

    Can run with:
    1. Fine-tuned local model (if available)
    2. Gemini API fallback (for demo without local model)
    """

    def __init__(self, model_path: str = "models/medical-specialist", use_local: bool = True):
        self.model = None
        self.tokenizer = None
        self.use_local = use_local
        self.model_path = model_path

        if use_local and os.path.exists(model_path):
            self._load_local_model(model_path)
        else:
            self.use_local = False

    def _load_local_model(self, model_path: str):
        """Load the fine-tuned model from disk."""
        try:
            from src.model import load_fine_tuned_model
            self.model, self.tokenizer = load_fine_tuned_model(
                adapter_path=model_path,
            )
            print(f"Loaded fine-tuned model from {model_path}")
        except Exception as e:
            print(f"Could not load local model: {e}")
            self.use_local = False

    def ask(self, question: str, max_new_tokens: int = 256) -> dict:
        """
        Ask the medical specialist a question.

        Returns dict with answer and metadata.
        """
        if self.use_local and self.model is not None:
            return self._ask_local(question, max_new_tokens)
        else:
            return self._ask_gemini(question)

    def _ask_local(self, question: str, max_new_tokens: int = 256) -> dict:
        """Generate answer using local fine-tuned model."""
        from src.evaluate import generate_response
        import time

        start = time.time()
        response = generate_response(self.model, self.tokenizer, question, max_new_tokens)
        elapsed = time.time() - start

        return {
            "answer": response,
            "model": "Fine-tuned TinyLlama (local)",
            "time_seconds": round(elapsed, 2),
            "source": "local",
        }

    def _ask_gemini(self, question: str) -> dict:
        """Fallback: use Gemini API with medical system prompt."""
        import time

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {
                "answer": "No model available. Set GOOGLE_API_KEY or provide fine-tuned model.",
                "model": "none",
                "time_seconds": 0,
                "source": "error",
            }

        import google.generativeai as genai
        genai.configure(api_key=api_key)

        system_prompt = (
            "You are a medical specialist AI assistant fine-tuned on medical Q&A data. "
            "Provide accurate, detailed medical information. Use proper medical terminology. "
            "Structure your responses clearly with headings and bullet points. "
            "Always recommend consulting a healthcare professional for personal medical advice."
        )

        model = genai.GenerativeModel(
            "gemini-2.0-flash-lite",
            system_instruction=system_prompt,
        )

        start = time.time()
        try:
            response = model.generate_content(question)
            answer = response.text
        except Exception as e:
            answer = f"Error: {str(e)}"

        elapsed = time.time() - start

        return {
            "answer": answer,
            "model": "Gemini 2.5 Flash (API fallback)",
            "time_seconds": round(elapsed, 2),
            "source": "gemini",
        }


def get_base_model_response(question: str) -> dict:
    """
    Get a response from the BASE (non-fine-tuned) model via Gemini.

    Uses a generic system prompt to simulate a non-specialized model.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {
            "answer": "No API key available.",
            "model": "none",
            "time_seconds": 0,
            "source": "error",
        }

    import google.generativeai as genai
    import time

    genai.configure(api_key=api_key)

    # Generic prompt — no medical specialization
    model = genai.GenerativeModel(
        "gemini-2.0-flash-lite",
        system_instruction="You are a helpful assistant. Keep responses brief and simple. Use everyday language, avoid technical jargon.",
    )

    start = time.time()
    try:
        response = model.generate_content(question)
        answer = response.text
    except Exception as e:
        answer = f"Error: {str(e)}"

    elapsed = time.time() - start

    return {
        "answer": answer,
        "model": "Generic LLM (not fine-tuned)",
        "time_seconds": round(elapsed, 2),
        "source": "gemini_base",
    }
