# Retina Disease App

An interactive Streamlit web app that classifies retinal OCT (Optical Coherence Tomography) scans into **eight disease categories** and explains every prediction.

## What it does

- **Upload an OCT scan** — the app predicts the diagnosis with a confidence score, top-3 probabilities, the affected retinal region, and a clinical note per class.
- **Explainable AI visualizations** — Grad-CAM, Grad-CAM++, and Eigen-CAM heatmaps overlaid on the scan, plus localization and activation-percentage readouts, so you can see *why* the model predicted what it did.
- **LLM-assisted reporting agent** — powered by Groq (LLaMA-3.3-70b) and grounded in the project's evaluation artifacts, it drafts diagnostic summaries and answers follow-up questions about each result ("Ask About This Result").
- **Orchestrated agents** — an orchestrator routes requests across vision, XAI, and reporting agents for a smooth end-to-end flow.

## Model

A fine-tuned **EfficientNetV2S** (TensorFlow/Keras, ImageNet-pretrained with the last 40 layers unfrozen) with a global average pooling + batch normalization + dropout (0.30) + softmax classification head. It is trained on **384x384x3 OCT images** from the **Retinal OCT-C8 dataset** (24,000 images across 8 classes: AMD, CNV, CSR, DME, DR, DRUSEN, MH, NORMAL — 18,400 train / 2,800 val / 2,800 test). The fine-tuned weights are fetched at runtime from Hugging Face (`Avinash0410/retina_APP`).

## Results

- Fine-tuning the backbone materially improves over the ImageNet baseline; the built-in reporting agent surfaces the fine-tuned accuracy summary, per-class F1 scores, and confusion-pair analysis for the held-out test set.
- Per-prediction outputs include predicted class, confidence, top-3 probabilities, three XAI heatmaps, affected region, localization, activation %, and a class-specific clinical note.
- Same EfficientNetV2S approach as our AICS 2026 paper (Paper 48, accepted for oral presentation at AICS 2026).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `GROQ_API_KEY` to enable the LLM reporting agent. A `Dockerfile` and `.streamlit` config are included for containerized/cloud deployment.
