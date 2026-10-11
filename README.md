# EHR-Matrix AI

A multi-modal sepsis risk prediction system, built end-to-end: data pipeline, custom-trained model, and a live dashboard. No external AI API is used anywhere — the risk scores come from a model trained from scratch.

**Live demo:** https://ehr-matrix-ai.vercel.app/

> Research prototype built as an independent portfolio project. Not a clinical decision tool.

---

## What it does

EHR-Matrix AI predicts the risk that an ICU patient will develop sepsis in the next 24 hours, using:
- **Hourly vitals** (heart rate, blood pressure, respiratory rate, temperature, SpO2)
- **Lab results** (lactate, WBC, creatinine, platelets, glucose), including how recently each was measured
- **Clinical notes**, embedded with ClinicalBERT

The risk score, vitals, labs, and notes are shown together on a dashboard, alongside a plain-language summary and a breakdown of what's driving the risk at any given hour.

## Why synthetic notes

The original plan used MIMIC-IV, which has real clinical notes. Credentialed access wasn't available as an independent student with no course, company, or lab affiliation, so the project uses the **open PhysioNet 2019 sepsis challenge dataset** instead, and generates synthetic clinical notes directly from each patient's own vitals and labs. The notes never see the sepsis label and never use the word "sepsis." This means the notes can't add information the numeric model doesn't already have — see the case study for the full discussion of this tradeoff.

## Architecture


```text
Vitals + Labs (Hourly) ──► LSTM ──────────────┐
                                              ├──► Fusion Layer ──► Sepsis Risk (0–1)
Clinical Notes ──────────► ClinicalBERT ──► MLP ┘
```


- **Numeric branch:** LSTM over 7 vitals + 3 static fields + 5 labs (32 features total, including "hours since last measured")
- **Text branch:** ClinicalBERT embedding of the most recent note at each hour, reduced through a small dense layer
- **Fusion:** both branches concatenated and passed through a final classifier

## Results

Evaluated on a held-out hospital the model never trained on (AUPRC, train / val / test):

| Model | Features | Train | Val | Test |
|---|---|---|---|---|
| Logistic regression | Vitals + labs | 0.134 | 0.125 | 0.073 |
| XGBoost | Vitals + labs | 0.254 | 0.193 | 0.102 |
| LSTM | Vitals + labs | 0.295 | 0.225 | 0.125 |
| **Fusion (LSTM + text)** | Vitals + labs + notes | 0.275 | 0.227 | **0.126** |

Full ablation and discussion in the case study.

## Repo structure


```text
ehr-project/
├── explore_sepsis_2019.py       # Initial dataset exploration
├── build_cache.py               # Fast .psv → pandas cache
├── build_modeling_table.py      # Hourly table, labels, train/val/test split
├── sanity_checks.py             # Label leakage and missingness checks
├── train_baseline.py            # Logistic regression and XGBoost baselines
├── train_lstm.py                # LSTM on vitals and laboratory measurements
├── generate_notes.py            # Synthetic clinical note generation
├── embed_notes.py               # ClinicalBERT embeddings (GPU)
├── train_fusion.py              # Numeric, text, and fusion model training
├── export_dashboard_data.py     # Runs trained model and exports dashboard JSON
└── dashboard/                   # Next.js dashboard (deployed separately)
```



## Running it yourself

Requires Python 3.10+, Node 18+, and the [PhysioNet 2019 sepsis dataset](https://physionet.org/content/challenge-2019/1.0.0/) (open access, no credentialing needed).

```bash
# 1. Build the data pipeline
python build_cache.py
python build_modeling_table.py
python sanity_checks.py          # optional, verifies no label leakage

# 2. Train baselines and the model
python train_baseline.py
python train_lstm.py
python generate_notes.py
python embed_notes.py            # GPU strongly recommended (Colab/Kaggle work)
python train_fusion.py --mode fusion

# 3. Generate dashboard data and run it
python export_dashboard_data.py
cd dashboard
npm install
npm run dev                      # http://localhost:3000
```

## Tech stack

Python, PyTorch, scikit-learn, XGBoost, Hugging Face Transformers (ClinicalBERT), Next.js, deployed on Vercel.

## Data and licensing

- Dataset: Reyna et al., "Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019," *Critical Care Medicine*, 2020. Open access, CC BY 4.0.
- Clinical notes in this project are synthetically generated and do not represent real patient documentation.
- This project is for educational/portfolio purposes only and is not validated for clinical use.

