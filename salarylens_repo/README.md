# 💰 SalaryLens India

**Real salary intelligence for the Indian job market — built on 177K+ verified data points.**

> Predicts P25 / P50 / P75 salary ranges for any role, company, city and experience level across India.  
> Built as a FAANG-prep portfolio project demonstrating end-to-end ML, NLP, and production deployment.

---

## Live Demo

Run on Google Colab → expose via ngrok → live in any browser. No cloud deployment needed.

---

## What it does

| Feature | Detail |
|---|---|
| **Salary Prediction** | P25 / P50 / P75 quantile range for any role + company + city + YoE |
| **Underpaid Check** | Compare your current CTC vs market median |
| **Company Compare** | Side-by-side salary ranges for up to 8 companies |
| **Salary Band Matrix** | 7 roles × 6 companies heatmap — adjustable by city and experience |
| **City Compare** | Same role across multiple Indian cities |
| **Growth Curve** | Salary progression from 0–15 years experience |
| **Negotiation Tips** | Context-aware advice based on company type and seniority |
| **Unknown Company Handling** | Web search + Claude AI enrichment; hard stop with similar suggestions if fully unknown |

---

## Architecture

```
Data Sources (177K+ rows)
├── Levels.fyi   (v3 / v4 / v5)  — verified tech salaries, weight 3.0x
├── Glassdoor    (30K sample)     — self-reported, weight 0.4x
└── Job Postings (100K sample)   — market demand signal, weight 0.5x

Feature Engineering
├── Role normalisation        (18 categories via regex rules)
├── Seniority extraction      (0–7 from title + level field)
├── Company classification    (157 named companies + keyword fallback)
├── City tier mapping         (Tier 1 / 2 / 3 — 17 cities)
├── Sentence-BERT embeddings  (384D → PCA 32D)
└── Skill flags               (40+ tech skills from job title)

Models (quantile regression blend)
├── LightGBM  P25 / P50 / P75   (60% weight)
├── CatBoost  P25 / P50 / P75   (40% weight)
└── Sort-based monotonicity      (guaranteed P25 ≤ P50 ≤ P75)

App
└── Streamlit (4 tabs) → ngrok tunnel → browser
```

---

## Model Performance

| Metric | Value | Target |
|---|---|---|
| P50 MAE | ₹5.96 LPA | — |
| P50 Calibration | 50.0% | 50% (perfect) |
| Range Accuracy (P25–P75) | 48.5% | 45–60% |
| Quantile Crossings | 0 | 0 |
| Sanity Check Pass Rate | 4/7 → improved post fixes | — |

**Known limitation:** High-end predictions (Google/Amazon senior roles, ₹70L+) are slightly compressed due to data sparsity above ₹70L in training data.

---

## Fixes Applied (v2 vs v1)

| Fix | Problem | Solution |
|---|---|---|
| Source reweighting | 56% job postings dominated training | Levels.fyi weight 3× (was 1×), job postings 0.5× (was 0.7×) |
| `has_yac` binary flag | `yac` was 90%+ imputed zeros — ghost signal | Replace with binary: did user report years at company? |
| `is_tier1_mnc` feature | Google/Amazon predictions ≈ generic MNC | Explicit flag for FAANG-level companies |
| Sort-based monotonicity | 16 quantile crossings in v1 | `np.sort` on stack — mathematically guaranteed 0 crossings |

---

## Quick Start

### 1. Upload data to Google Drive

Create a folder called `salary_prediction_files` in your Google Drive and upload:

```
levels_v5_data.csv
levels_v4_data.csv
levels_v3_data.csv
glassdoor_salary_data.csv
jobposting_salary_data.csv
```

Data sources: [Levels.fyi](https://www.levels.fyi) · [Glassdoor](https://www.glassdoor.co.in) · [Job Postings dataset]

### 2. Open notebook in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/salarylens-india/blob/main/SalaryLens_India.ipynb)

### 3. Run cells in order

| Cell | What it does | Time |
|---|---|---|
| Cell 1 | Install packages | ~2 min |
| Cell 2 | Mount Drive + config + load data | ~5 min |
| Cell 3 | Feature engineering | ~3 min |
| Cell 4 | Train LightGBM + CatBoost | ~15 min |
| Cell 5 | Evaluate + save model | ~2 min |
| Cell 6 | Inference sanity check | ~1 min |
| Cell 7 | Launch Streamlit app | ~1 min |

### 4. Get ngrok token

Sign up free at [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).  
Paste your token in Cell 7 where it says `PASTE_YOUR_TOKEN_HERE`.

---

## Project Structure

```
salarylens-india/
├── SalaryLens_India.ipynb   # Main notebook — full pipeline
├── streamlit_app.py          # Extracted Streamlit app (for reference)
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes model artifacts, data, secrets
└── README.md                 # This file
```

**Not in repo (too large / sensitive):**
- `*.pkl` model artifacts → stored in Google Drive
- Raw CSV data files → stored in Google Drive
- ngrok token → set locally in Cell 7

---

## Company Coverage

The model handles **157 named companies** across 5 categories:

| Category | Companies | Examples |
|---|---|---|
| MNC Product | 31 + 20 extended | Google, Amazon, Microsoft, Stripe, Datadog |
| Indian Product | 19 + 22 extended | Flipkart, Razorpay, Zerodha, Nykaa |
| IT Services | 14 + 5 extended | Infosys, TCS, Wipro, Cognizant |
| MNC Finance | 12 + 9 extended | Goldman Sachs, JPMorgan, HDFC |
| Consulting | 11 + 5 extended | Deloitte, McKinsey, Accenture |

For unknown companies, the app runs a 3-layer enrichment pipeline:
1. Local lookup (157 named + keyword rules)
2. DuckDuckGo web search (free, no API key)
3. Claude AI classification (requires Anthropic API key)

If all three fail, the app shows **"No data available"** with similar company suggestions — no prediction is made.

---

## Tech Stack

- **ML:** LightGBM · CatBoost · scikit-learn
- **NLP:** Sentence-BERT (all-MiniLM-L6-v2) · PCA
- **Explainability:** SHAP
- **App:** Streamlit
- **Tunnel:** ngrok
- **Enrichment:** DuckDuckGo API · Anthropic Claude API
- **Platform:** Google Colab + Google Drive

---

## Interview / Portfolio Notes

This project demonstrates:

- **Quantile regression** for uncertainty-aware prediction (P25/P50/P75)
- **Ensemble blending** (LightGBM + CatBoost, 60/40)
- **Semantic embeddings** for job title normalisation
- **Source quality weighting** in training data
- **SHAP explainability** for model transparency
- **Production-grade** unknown-company handling with graceful degradation
- **Real data** (177K+ records from 3 independent sources)

---

## Disclaimer

Salary predictions are for reference only. Actual compensation varies based on negotiation, skills, performance and market conditions. Data sourced from public platforms as of 2024–25.

---

*Built by Sanya*
