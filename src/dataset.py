"""
Dataset preparation for medical Q&A fine-tuning.
"""

import json
import os
from datasets import load_dataset


def load_medical_qa(max_samples: int = 5000) -> dict:
    """
    Load medical Q&A dataset from HuggingFace.

    Uses keivalya/MedQuad-MedicalQnADataset — 47K medical Q&A pairs
    from NIH websites covering 37 medical topics.
    """
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

    # Shuffle and limit
    dataset = dataset.shuffle(seed=42)
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    return dataset


def format_for_training(example: dict) -> dict:
    """
    Format a single example into the chat template.

    TinyLlama chat format:
    <|user|>
    {question}</s>
    <|assistant|>
    {answer}</s>
    """
    question = example.get("Question", "")
    answer = example.get("Answer", "")

    # Chat template for TinyLlama
    text = (
        f"<|user|>\n"
        f"You are a medical specialist. Answer this question accurately and clearly.\n\n"
        f"{question}</s>\n"
        f"<|assistant|>\n"
        f"{answer}</s>"
    )

    return {"text": text}


def prepare_dataset(max_samples: int = 5000, val_ratio: float = 0.1):
    """
    Full dataset preparation pipeline.

    Returns train and validation datasets formatted for fine-tuning.
    """
    print(f"Loading medical Q&A dataset (max {max_samples} samples)...")
    dataset = load_medical_qa(max_samples)
    print(f"Loaded {len(dataset)} samples")

    # Format for training
    formatted = dataset.map(format_for_training)

    # Train/val split
    split = formatted.train_test_split(test_size=val_ratio, seed=42)
    train_data = split["train"]
    val_data = split["test"]

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    return train_data, val_data


def create_sample_questions() -> list[dict]:
    """
    Create sample questions for demo/evaluation.

    These are manually curated to test different medical knowledge areas.
    """
    samples = [
        {
            "question": "What are the common symptoms of Type 2 diabetes?",
            "topic": "Endocrinology",
        },
        {
            "question": "How is hypertension diagnosed and what are the treatment options?",
            "topic": "Cardiology",
        },
        {
            "question": "What causes iron deficiency anemia and how is it treated?",
            "topic": "Hematology",
        },
        {
            "question": "What are the warning signs of a stroke?",
            "topic": "Neurology",
        },
        {
            "question": "How does asthma affect the airways and what triggers attacks?",
            "topic": "Pulmonology",
        },
        {
            "question": "What is the difference between Crohn's disease and ulcerative colitis?",
            "topic": "Gastroenterology",
        },
        {
            "question": "What are the risk factors for developing osteoporosis?",
            "topic": "Rheumatology",
        },
        {
            "question": "How is pneumonia diagnosed and treated?",
            "topic": "Infectious Disease",
        },
        {
            "question": "What are the stages of chronic kidney disease?",
            "topic": "Nephrology",
        },
        {
            "question": "What lifestyle changes help manage high cholesterol?",
            "topic": "Preventive Medicine",
        },
    ]
    return samples


def save_sample_questions(output_path: str = "data/sample_questions.json"):
    """Save sample questions to JSON for the demo app."""
    samples = create_sample_questions()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} sample questions to {output_path}")
    return samples


if __name__ == "__main__":
    # Quick test
    save_sample_questions()
    train, val = prepare_dataset(max_samples=100)
    print(f"\nSample formatted text:\n{train[0]['text'][:500]}...")
