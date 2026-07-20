"""
reporting_agent.py

Loads the saved project artifacts (metrics, confusion pairs, summary text) once
at import time, and exposes ask_reporting_agent(question) which either answers
from a small set of deterministic, keyword-routed summaries, or falls back to
Groq/LLaMA for open-ended questions — always grounded in the loaded artifacts,
never given free rein to invent numbers.
"""

import os


import pandas as pd
from dotenv import load_dotenv
from groq import Groq

#from config import REPORTS_DIR


#SUMMARY_PATH = REPORTS_DIR / "final_summary.txt"
#COMPARISON_PATH = REPORTS_DIR / "baseline_vs_finetuned_comparison.csv"
#METRICS_PATH = REPORTS_DIR / "finetuned_test_classification_metrics.csv"
#CONFUSION_PATH = REPORTS_DIR / "top_confusion_pairs.csv"

s#ummary_text = SUMMARY_PATH.read_text(encoding="utf-8") if SUMMARY_PATH.exists() else ""
#comparison_df = pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else None
#metrics_df = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None
#confusion_df = pd.read_csv(CONFUSION_PATH) if CONFUSION_PATH.exists() else None


load_dotenv()  
client = Groq(api_key=os.getenv("GROQ_API_KEY"))    


def ask_groq(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are an ophthalmology AI assistant helping explain retinal OCT classification results.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content



def percent(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)


def get_accuracy_summary():
    if comparison_df is None:
        return "The baseline vs fine-tuned comparison report is not available."
    df = comparison_df.copy()
    baseline_row = df[df["model"].astype(str).str.lower().str.contains("baseline")]
    fine_row = df[df["model"].astype(str).str.lower().str.contains("fine")]
    if baseline_row.empty or fine_row.empty:
        return df.to_string(index=False)
    baseline_row, fine_row = baseline_row.iloc[0], fine_row.iloc[0]
    return (
        f"The baseline model achieved {percent(baseline_row['test_accuracy'])} test accuracy "
        f"and {percent(baseline_row['val_accuracy'])} validation accuracy. "
        f"After fine-tuning, the model improved to {percent(fine_row['test_accuracy'])} test accuracy "
        f"and {percent(fine_row['val_accuracy'])} validation accuracy."
    )


def get_best_worst_classes():
    if metrics_df is None:
        return "The fine-tuned class metrics report is not available."
    df = metrics_df.copy()
    class_col = "class_name" if "class_name" in df.columns else df.columns[0]
    f1_col = "f1_score" if "f1_score" in df.columns else "f1-score"
    if f1_col not in df.columns:
        return "The F1-score column was not found."
    valid_classes = ["amd", "cnv", "csr", "dme", "dr", "drusen", "mh", "normal"]
    valid_df = df[df[class_col].astype(str).str.lower().isin(valid_classes)].copy()
    valid_df[f1_col] = pd.to_numeric(valid_df[f1_col], errors="coerce")
    valid_df = valid_df.dropna(subset=[f1_col])
    if valid_df.empty:
        return "No valid class-wise F1-score rows found."
    best = valid_df.sort_values(f1_col, ascending=False).head(3)
    worst = valid_df.sort_values(f1_col, ascending=True).head(3)
    best_text = ", ".join([f"{row[class_col]} ({row[f1_col]:.3f} F1)" for _, row in best.iterrows()])
    worst_text = ", ".join([f"{row[class_col]} ({row[f1_col]:.3f} F1)" for _, row in worst.iterrows()])
    return f"Strongest classes: {best_text}. Weakest classes: {worst_text}."


def get_confusion_summary():
    if confusion_df is None:
        return "Confusion-pair report not available."
    text = confusion_df.head(5).to_string(index=False)
    return f"Top confusion pairs:\n{text}"


def get_xai_summary():
    return "Grad-CAM, Grad-CAM++, and Eigen-CAM were used for explainability. They generate heatmaps showing which retinal regions influenced the prediction."


def get_dataset_summary():
    return "The project used the Retinal OCT-C8 dataset with 24,000 images across 8 classes (AMD, CNV, CSR, DME, DR, DRUSEN, MH, NORMAL). Each class has 3,000 images."


def get_split_summary():
    return "Each class has 2,300 training, 350 validation, and 350 test images. Totals: 18,400 train, 2,800 val, 2,800 test."


def get_limitations_summary():
    return "The model is a research prototype, not ready for clinical use. It needs external validation and regulatory approval."


def get_goal_summary():
    return "The goal is to build an explainable deep-learning pipeline for classifying retinal OCT images into 8 disease categories with XAI heatmaps."


def get_project_summary():
    return "Explainable retinal OCT classification pipeline. " + get_accuracy_summary() + " Includes confusion analysis and XAI heatmaps."



def ask_reporting_agent(question, prediction_context=None):
    """
    Answer questions about the project or the current OCT prediction.

    prediction_context contains the most recent result produced
    by the Vision Agent. This allows follow-up questions such as:
        - What precautions should be taken?
        - What does this condition mean?
        - What should I do next?

    to be answered in the context of the current prediction.
    """
    
    #print(prediction_context)
    q = question.lower().strip()

    

    prediction_keywords = [
    "precaution",
    "precautions",
    "care",
    "treatment",
    "treat",
    "medicine",
    "medication",
    "manage",
    "management",
    "prevent",
    "prevention",
    "what should i do",
    "next step",
    "condition",
    "disease",
    "diagnosis",
    "symptom",
    "symptoms",

    
    "region",
    "affected",
    "where",
    "localization",
    "location",
    "activation",
]

    is_prediction_question = any(
        keyword in q for keyword in prediction_keywords
    )

    if prediction_context and is_prediction_question:

        predicted_class = prediction_context.get(
        "predicted_class",
        "Unknown",
    )

        confidence = prediction_context.get(
        "confidence",
        "Unknown",
    )

        affected_region = prediction_context.get(
        "region",
        "Not available",
    )

        localization = prediction_context.get(
        "localization",
        "Not available",
    )

        activation_pct = prediction_context.get(
        "activation_pct",
        "Not available",
    )

        clinical_note = prediction_context.get(
        "clinical_note",
        "Not available",
    )

    
    if any(word in q for word in ["region", "affected", "where", "location"]):
        return (
            f"The affected retinal region is **{affected_region}**.\n\n"
            f"Localization: **{localization}**\n\n"
            f"Activation covers approximately **{activation_pct}%** of the OCT image."
        )

    
        prompt = f"""
You are an AI assistant for a research prototype that analyses
retinal OCT images.

The most recent OCT image analysis produced the following result:

Predicted retinal condition:
{predicted_class}

Model confidence:
{confidence}

Affected retinal region:
{affected_region}

Localization:
{localization}

Activation:
{activation_pct}% of the OCT image

Clinical information:
{clinical_note}

The user asked:
"{question}"

Answer the user's question directly and specifically in the context
of the predicted retinal condition.

Important instructions:

- Do not repeat the entire OCT prediction.
- Do not discuss model accuracy, F1 scores, confusion matrices,
  dataset statistics, or training performance unless the user
  specifically asks about them.
- If the user asks about precautions, treatment, management, or
  next steps, provide general educational information relevant
  to the predicted condition.
- Do not prescribe medication or give a definitive medical diagnosis.
- Explain that the AI prediction is a research result and should
  be clinically confirmed by a qualified ophthalmologist.
- Recommend appropriate professional medical evaluation when relevant.
- Keep the answer concise and directly related to the user's question.
"""

        return ask_groq(prompt)

    
    if any(
        p in q
        for p in [
            "main goal",
            "goal",
            "objective",
            "purpose",
        ]
    ):
        return get_goal_summary()

    
    elif any(
        w in q
        for w in [
            "best",
            "worst",
            "f1",
            "metric",
            "metrics",
            "performed",
        ]
    ):
        return get_best_worst_classes()

    

    elif any(
        w in q
        for w in [
            "accuracy",
            "performance",
            "result",
            "baseline",
            "fine-tuned",
            "compare",
        ]
    ):
        return get_accuracy_summary()

    
    elif any(
        w in q
        for w in [
            "confusion",
            "mistake",
            "misclassified",
            "error",
        ]
    ):
        return get_confusion_summary()

   
    elif any(
        w in q
        for w in [
            "grad",
            "cam",
            "xai",
            "heatmap",
            "explainable",
        ]
    ):
        return get_xai_summary()

    
    elif any(
        w in q
        for w in [
            "clinical",
            "deploy",
            "limitation",
        ]
    ):
        return get_limitations_summary()

    
    elif any(
        w in q
        for w in [
            "split",
            "training",
            "validation",
            "testing",
        ]
    ):
        return get_split_summary()

    
    elif any(
        w in q
        for w in [
            "dataset",
            "data",
            "classes",
            "images",
        ]
    ):
        return get_dataset_summary()

    
    elif any(
        w in q
        for w in [
            "summary",
            "summarize",
            "overview",
        ]
    ):
        return get_project_summary()

    
    prediction_info = ""



    if prediction_context:

        prediction_info = f"""
Current OCT Prediction:

Predicted condition:
{prediction_context.get("predicted_class", "Unknown")}

Confidence:
{prediction_context.get("confidence", "Unknown")}

Affected region:
{prediction_context.get("affected_region", "Not available")}

Clinical note:
{prediction_context.get("clinical_note", "Not available")}
"""

    context = f"""
You are an AI assistant for an explainable retinal OCT
classification research project.

{prediction_info}

Project Summary:
{summary_text}

Model Accuracy:
{get_accuracy_summary()}

Class Metrics:
{get_best_worst_classes()}

Confusion Analysis:
{get_confusion_summary()}

Dataset:
{get_dataset_summary()}

User Question:
{question}

Instructions:

Answer the user's question directly.

If the question relates to the current OCT prediction, prioritize
the current prediction context.

If the question relates to model performance, use the project
metrics provided above.

Do not invent numerical results.

Do not repeat unrelated project statistics.

The OCT classifier is a research prototype and its predictions
are not a confirmed medical diagnosis.
"""

    return ask_groq(context)