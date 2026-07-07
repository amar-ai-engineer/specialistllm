"""
Evaluation — compare base model vs fine-tuned model.
"""

import json
import os
import time


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the model."""
    import torch

    prompt = (
        f"<|user|>\n"
        f"You are a medical specialist. Answer this question accurately and clearly.\n\n"
        f"{question}</s>\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def compare_models(
    base_model,
    base_tokenizer,
    ft_model,
    ft_tokenizer,
    questions: list[dict],
) -> list[dict]:
    """
    Run the same questions through base and fine-tuned models.

    Returns comparison results for each question.
    """
    results = []

    for q in questions:
        question = q["question"]
        topic = q.get("topic", "General")

        print(f"  Evaluating: {question[:60]}...")

        # Base model response
        start = time.time()
        base_response = generate_response(base_model, base_tokenizer, question)
        base_time = time.time() - start

        # Fine-tuned model response
        start = time.time()
        ft_response = generate_response(ft_model, ft_tokenizer, question)
        ft_time = time.time() - start

        results.append({
            "question": question,
            "topic": topic,
            "base_response": base_response,
            "finetuned_response": ft_response,
            "base_time_seconds": round(base_time, 2),
            "finetuned_time_seconds": round(ft_time, 2),
        })

    return results


def evaluate_response_quality(response: str) -> dict:
    """
    Basic heuristic evaluation of a response.

    Not a replacement for human evaluation, but useful for quick checks.
    """
    # Length check
    word_count = len(response.split())

    # Structure check (does it have organized content?)
    has_structure = any(
        marker in response
        for marker in ["1.", "2.", "- ", "* ", ":", "First", "Second"]
    )

    # Medical terminology presence (basic check)
    medical_terms = [
        "diagnosis", "treatment", "symptoms", "medication", "therapy",
        "chronic", "acute", "prognosis", "clinical", "patient",
        "blood pressure", "cholesterol", "glucose", "inflammation",
    ]
    medical_term_count = sum(1 for term in medical_terms if term.lower() in response.lower())

    # Disclaimer/safety check
    has_disclaimer = any(
        phrase in response.lower()
        for phrase in ["consult", "doctor", "healthcare", "medical professional", "seek"]
    )

    return {
        "word_count": word_count,
        "has_structure": has_structure,
        "medical_terms_found": medical_term_count,
        "has_safety_disclaimer": has_disclaimer,
        "quality_score": min(10, word_count // 20 + medical_term_count + (2 if has_structure else 0)),
    }


def save_comparison_results(results: list[dict], output_path: str = "data/comparison_results.json"):
    """Save comparison results for the Streamlit app to display."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Add quality scores
    for r in results:
        r["base_quality"] = evaluate_response_quality(r["base_response"])
        r["finetuned_quality"] = evaluate_response_quality(r["finetuned_response"])

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved comparison results to {output_path}")
    return results


def create_demo_results() -> list[dict]:
    """
    Create pre-computed demo results for the Streamlit app.

    These simulate what a base vs fine-tuned comparison looks like.
    In real usage, these come from actual model inference.
    """
    results = [
        {
            "question": "What are the common symptoms of Type 2 diabetes?",
            "topic": "Endocrinology",
            "base_response": "Type 2 diabetes is a condition where your body doesn't use insulin properly. Some symptoms include feeling thirsty, going to the bathroom a lot, and feeling tired. You should see a doctor if you have these symptoms.",
            "finetuned_response": "Type 2 diabetes presents with several characteristic symptoms:\n\n1. **Polyuria** (frequent urination) — excess glucose is excreted through urine\n2. **Polydipsia** (increased thirst) — fluid loss from polyuria triggers thirst\n3. **Polyphagia** (increased hunger) — cells can't absorb glucose properly\n4. **Unexplained weight loss** — despite increased appetite\n5. **Fatigue** — cells are deprived of glucose energy\n6. **Blurred vision** — high glucose affects the lens\n7. **Slow-healing wounds** — impaired blood flow and immune function\n8. **Numbness/tingling in extremities** — early peripheral neuropathy\n\nMany patients are asymptomatic in early stages, which is why screening (fasting glucose or HbA1c) is recommended for adults over 45 or those with risk factors (obesity, family history, sedentary lifestyle).\n\nConsult your healthcare provider for proper diagnosis and management.",
            "base_quality": {"word_count": 42, "has_structure": False, "medical_terms_found": 1, "has_safety_disclaimer": True, "quality_score": 5},
            "finetuned_quality": {"word_count": 128, "has_structure": True, "medical_terms_found": 6, "has_safety_disclaimer": True, "quality_score": 10},
        },
        {
            "question": "How is hypertension diagnosed and what are the treatment options?",
            "topic": "Cardiology",
            "base_response": "Hypertension means high blood pressure. Doctors check it with a blood pressure cuff. If it's high, they might give you medicine or tell you to eat less salt and exercise more.",
            "finetuned_response": "**Diagnosis of Hypertension:**\n\nBlood pressure is measured in mmHg with two values:\n- **Normal**: <120/80 mmHg\n- **Elevated**: 120-129/<80 mmHg\n- **Stage 1 HTN**: 130-139/80-89 mmHg\n- **Stage 2 HTN**: ≥140/≥90 mmHg\n\nDiagnosis requires elevated readings on at least 2-3 separate occasions. Ambulatory blood pressure monitoring (ABPM) is the gold standard to rule out white-coat hypertension.\n\n**Treatment Options:**\n\n*Lifestyle modifications (first-line for all stages):*\n- DASH diet (rich in fruits, vegetables, low-fat dairy)\n- Sodium restriction (<2,300 mg/day, ideally <1,500 mg)\n- Regular aerobic exercise (150 min/week)\n- Weight management (BMI <25)\n- Limit alcohol intake\n\n*Pharmacological therapy:*\n- **ACE inhibitors** (lisinopril, enalapril) — first-line, especially with diabetes/CKD\n- **ARBs** (losartan, valsartan) — alternative to ACE-I\n- **Calcium channel blockers** (amlodipine) — good for older adults\n- **Thiazide diuretics** (hydrochlorothiazide) — effective and inexpensive\n\nMedication choice depends on comorbidities, age, and patient-specific factors. Regular monitoring and follow-up are essential.",
            "base_quality": {"word_count": 36, "has_structure": False, "medical_terms_found": 2, "has_safety_disclaimer": False, "quality_score": 4},
            "finetuned_quality": {"word_count": 168, "has_structure": True, "medical_terms_found": 8, "has_safety_disclaimer": False, "quality_score": 10},
        },
        {
            "question": "What are the warning signs of a stroke?",
            "topic": "Neurology",
            "base_response": "A stroke happens when blood flow to the brain is blocked. Warning signs include face drooping, arm weakness, and speech difficulty. Call 911 right away if you notice these signs. Time is very important.",
            "finetuned_response": "The warning signs of a stroke are best remembered using the **FAST** acronym:\n\n- **F — Face drooping**: One side of the face droops or feels numb. Ask the person to smile — is it uneven?\n- **A — Arm weakness**: One arm feels weak or numb. Ask them to raise both arms — does one drift downward?\n- **S — Speech difficulty**: Speech is slurred or hard to understand. Ask them to repeat a simple sentence.\n- **T — Time to call emergency services**: If any of these signs are present, call 911 immediately. Note the time symptoms started — this determines treatment eligibility.\n\n**Additional warning signs:**\n- Sudden severe headache with no known cause\n- Sudden confusion or trouble understanding\n- Sudden vision problems in one or both eyes\n- Sudden dizziness, loss of balance, or coordination\n- Sudden numbness (especially on one side of the body)\n\n**Critical timing**: Thrombolytic therapy (tPA) is most effective within 4.5 hours of symptom onset. Every minute matters — approximately 1.9 million neurons are lost per minute during a stroke.\n\nSeek emergency medical attention immediately if you observe any of these signs.",
            "base_quality": {"word_count": 40, "has_structure": False, "medical_terms_found": 1, "has_safety_disclaimer": False, "quality_score": 3},
            "finetuned_quality": {"word_count": 175, "has_structure": True, "medical_terms_found": 5, "has_safety_disclaimer": True, "quality_score": 10},
        },
        {
            "question": "What causes iron deficiency anemia and how is it treated?",
            "topic": "Hematology",
            "base_response": "Iron deficiency anemia happens when you don't have enough iron. You might feel tired and weak. Eating iron-rich foods like red meat and spinach can help. Your doctor might also give you iron pills.",
            "finetuned_response": "**Iron deficiency anemia (IDA)** is the most common nutritional deficiency worldwide, affecting approximately 1.2 billion people.\n\n**Causes:**\n1. **Inadequate dietary intake** — common in vegetarians, elderly, and developing countries\n2. **Blood loss** — menstruation (most common cause in premenopausal women), GI bleeding (most common in men and postmenopausal women), ulcers, colorectal cancer\n3. **Malabsorption** — celiac disease, inflammatory bowel disease, gastric bypass surgery, H. pylori infection\n4. **Increased demand** — pregnancy, growth periods in children and adolescents\n\n**Diagnosis:**\n- Low serum ferritin (<30 ng/mL) — most specific marker\n- Low serum iron, elevated TIBC (total iron-binding capacity)\n- Low hemoglobin/hematocrit\n- Microcytic, hypochromic red blood cells on blood smear\n\n**Treatment:**\n- **Oral iron supplementation**: Ferrous sulfate 325mg, 1-3 times daily on empty stomach. Vitamin C enhances absorption.\n- **IV iron**: For patients who can't tolerate oral iron or have malabsorption (ferric carboxymaltose, iron sucrose)\n- **Treat underlying cause**: GI workup for unexplained IDA in men or postmenopausal women\n- **Dietary changes**: Red meat, legumes, fortified cereals, dark leafy greens\n\nResponse to treatment is monitored with reticulocyte count (rises in 1 week) and hemoglobin (normalizes in 6-8 weeks). Continue iron for 3-6 months after normalization to replenish stores.",
            "base_quality": {"word_count": 38, "has_structure": False, "medical_terms_found": 2, "has_safety_disclaimer": True, "quality_score": 4},
            "finetuned_quality": {"word_count": 198, "has_structure": True, "medical_terms_found": 7, "has_safety_disclaimer": False, "quality_score": 10},
        },
        {
            "question": "What is the difference between Crohn's disease and ulcerative colitis?",
            "topic": "Gastroenterology",
            "base_response": "Crohn's disease and ulcerative colitis are both types of inflammatory bowel disease. Crohn's can affect any part of the digestive tract while ulcerative colitis only affects the colon. Both cause stomach pain and diarrhea.",
            "finetuned_response": "Crohn's disease (CD) and ulcerative colitis (UC) are both inflammatory bowel diseases (IBD), but they differ in several important ways:\n\n| Feature | Crohn's Disease | Ulcerative Colitis |\n|---------|----------------|--------------------|\n| **Location** | Any part of GI tract (mouth to anus), most common in terminal ileum | Colon and rectum only |\n| **Pattern** | Skip lesions (patches of inflammation with healthy areas between) | Continuous inflammation starting from rectum |\n| **Depth** | Transmural (full thickness of bowel wall) | Mucosa and submucosa only |\n| **Complications** | Fistulas, strictures, abscesses | Toxic megacolon, increased colorectal cancer risk |\n| **Bloody diarrhea** | Less common | Very common (hallmark symptom) |\n| **Smoking** | Worsens disease | May be protective (but don't smoke) |\n| **Surgery** | Not curative; recurrence common | Colectomy is curative |\n\n**Shared features:**\n- Chronic, relapsing-remitting course\n- Extraintestinal manifestations (arthritis, skin lesions, eye inflammation)\n- Similar medications (aminosalicylates, corticosteroids, immunomodulators, biologics)\n- Peak onset: 15-35 years old\n\n**Diagnosis** involves colonoscopy with biopsy, imaging (CT/MRI enterography), and lab work (CRP, ESR, fecal calprotectin). In ~10% of cases, distinction between CD and UC is unclear (indeterminate colitis).",
            "base_quality": {"word_count": 38, "has_structure": False, "medical_terms_found": 3, "has_safety_disclaimer": False, "quality_score": 5},
            "finetuned_quality": {"word_count": 185, "has_structure": True, "medical_terms_found": 7, "has_safety_disclaimer": False, "quality_score": 10},
        },
    ]

    return results
