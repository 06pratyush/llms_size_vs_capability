<div align="center">

# 🧠 LLM Size vs Capability

### Does a bigger language model mean a better one? What should you actually run on a 4, 8 or 16 GB device?

A data analysis of **4,486 open LLMs** from the Hugging Face **Open LLM Leaderboard**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557C)
![seaborn](https://img.shields.io/badge/seaborn-4C72B0)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uwvLyJdW4ZVIPZ5Zl2DY2NIHiNiDj8X2)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Data-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![Models](https://img.shields.io/badge/models%20analysed-4%2C486-blue)
![Figures](https://img.shields.io/badge/figures-30-green)

<img src="figures/fig07_size_vs_score_main.png" alt="Model size against leaderboard score" width="900">

</div>

---

## 📌 TL;DR

> **"Pick the biggest model that fits" is the wrong rule. Within your memory budget, pick the best-scoring model that fits.**
> On average that gains **+12 leaderboard points**, which is about what you would get from a model **4–18× bigger**, with no extra hardware.

| | Key number | What it means |
|---|---|---|
| 📈 | **+13.8 pts** per 10× size | Bigger does help, but slowly |
| 🎯 | **40%** | Share of score differences explained by size alone |
| 🎲 | **8.4 pts** | Typical gap between a model's score and what its size predicts (worth a ~4× bigger model) |
| 🌲 | **75%** | Variance explained by a random forest using size + training type + blueprint + date |
| ⏳ | **~17 months** | How long it took 4 GB-class models to match 30B+ models |
| ⚡ | **~3×** energy per 10× size | Size costs energy much faster than it adds score |
| 👍 | **0.09** | Rank correlation between Hugging Face likes and score, i.e. popularity tells you almost nothing |

---

## 🗺️ What happens to the data

```mermaid
flowchart LR
    A["🤗 Hugging Face<br/>open-llm-leaderboard/contents<br/><b>4,576 rows</b>"] --> B["🧹 Cleaning<br/>−10 no size/score<br/>−1 flagged<br/>−79 duplicates"]
    B --> C["✅ <b>4,486 models</b><br/>leaderboard_clean.csv"]
    C --> D["🛠️ Feature engineering<br/>log size · 4/16-bit memory<br/>device fit · size band<br/>family · date · score/GB"]
    D --> E["📐 Size-only regression<br/>expected_score<br/>beat_curve = actual − expected"]
    E --> F["🌲 Random forest<br/>all traits · permutation importance"]
    F --> G["📊 30 figures<br/>+ findings report<br/>+ interactive explorer"]
    G --> H["🎯 Recommendation<br/>best model per<br/>4 / 8 / 16 GB"]
```

### Step by step

1. **Collect.** The raw table comes from the Hugging Face dataset [`open-llm-leaderboard/contents`](https://huggingface.co/datasets/open-llm-leaderboard/contents): one row per evaluated model, with six benchmark scores, parameter count, architecture, licence, likes and CO₂ cost.
2. **Clean.** 4,576 rows become **4,486 models** after removing 10 with no valid size or score, 1 flagged model and 79 duplicates. **2,823** of them can still be downloaded from the Hub.
3. **Normalise.** Every score is on the leaderboard's 0–100 scale, where 0 means random guessing and 100 means perfect. Model types are grouped into *Pretrained, Fine-tuned, Chat-tuned, Merge, Other*, and architectures into blueprint families (Llama, Qwen, Mistral, Gemma, …).
4. **Engineer features.**
   - `log_params`: size on a ×10 scale, because capability grows with the *multiple* of size, not the raw number
   - `footprint_4bit_gb` = 0.55 GB × params (B), plus `footprint_16bit_gb`
   - `fits_4gb / fits_8gb / fits_16gb`: a model may use 80% of device memory, so the limits are ≤5.8B, ≤11.6B and ≤23.3B parameters
   - `size_band`: Tiny ≤1B · Small 1–3.5B · Medium 3.5–9.5B · Large 9.5–16B · X-Large 16–36B · Huge >36B
   - `score_per_gb`, `model_date`, `years_since_2023`, `family_group`, `precision_group`
5. **Model size alone.** A straight-line fit, `score = a + b·log10(params)`, gives `expected_score`. The residual `beat_curve` (actual − expected) measures how much a model over- or under-performs for its size.
6. **Explain the rest.** Box plots, 95% confidence intervals and a **random forest** (trained on 75% of models, tested on the held-out 25%) measure how much training type, blueprint, date and yes/no traits add beyond size.
7. **Decide.** Using only models that can still be downloaded, find the **efficiency frontier** and the best pick for each device budget.

---

## 🔍 Findings at a glance

### 1️⃣ Size matters, but explains only 40%

<table>
<tr>
<td width="50%"><img src="figures/fig08_density_hexbin.png" alt="Density hexbin"></td>
<td width="50%"><img src="figures/fig09_violin_size_groups.png" alt="Violin by size group"></td>
</tr>
<tr>
<td align="center"><sub>Most models pile up at 6–12B, and scores there still range from about 5 to 34</sub></td>
<td align="center"><sub>Neighbouring size groups overlap heavily</sub></td>
</tr>
</table>

- Every **10× increase in size adds ~13.8 points**.
- Among 7–8B models alone, scores run from **7 to 34** (middle 90%).
- **60%** of Large (9.5–16B) models beat the *typical* X-Large (16–36B) model.

### 2️⃣ Choosing well at a given size is worth a ~4× bigger model

<table>
<tr>
<td width="50%"><img src="figures/fig11_beat_the_curve_histogram.png" alt="Beat the curve histogram"></td>
<td width="50%"><img src="figures/fig12_beat_curve_by_type.png" alt="Beat the curve by model type"></td>
</tr>
</table>

- A typical model lands **8.4 points** away from its size prediction.
- **Merges** sit highest (+1.6 median) and **base models** lowest (−6.4). Extra training is a strong lever.
- **Qwen** models beat their size by **+3.5** on average, while **GPT-NeoX** models fall **−13.0** short.

### 3️⃣ What the machine-learning model says

<table>
<tr>
<td width="33%"><img src="figures/fig15_model_comparison.png" alt="Model comparison"></td>
<td width="33%"><img src="figures/fig16_feature_importance.png" alt="Feature importance"></td>
<td width="33%"><img src="figures/fig17_predicted_vs_actual.png" alt="Predicted vs actual"></td>
</tr>
</table>

| Predictor (held-out test set) | R² |
|---|---|
| Straight line, size only | 38% |
| Random forest, size only | 66% |
| Random forest, **all traits** | **75%** |

Permutation importance: **Size 72%**, When it was made 13%, Model type 4%, Blueprint 4%, Chat template 3%, Times re-trained 2%. The forest's mean absolute error is **3.5 points**, against 8.4 for the size-only line.

### 4️⃣ Some skills need size and others need training

<table>
<tr>
<td width="50%"><img src="figures/fig19_skills_vs_size.png" alt="Skills vs size"></td>
<td width="50%"><img src="figures/fig21_radar_profiles.png" alt="Radar profiles"></td>
</tr>
</table>

| Benchmark | Skill | Points per 10× size | Depends on size |
|---|---|---|---|
| **BBH** | Hard multi-step reasoning | +21.2 | 🔴 Most (47%) |
| **MMLU-PRO** | Broad knowledge | +19.2 | 🔴 High |
| **IFEval** | Following exact instructions | +17.4 | 🟢 Least (18%) |
| **MATH Lvl 5** | Competition maths | +12.8 | 🟠 Medium |
| **MuSR** | Long-story reasoning | +6.5 | 🟢 Low |
| **GPQA** | PhD-level science | +5.9 | 🟢 Low |

> Two models with near-identical overall scores (27.1 vs 27.5) differ by **79 points** on IFEval. Always check the benchmark that matches your task.

### 5️⃣ Small models are catching up

<table>
<tr>
<td width="50%"><img src="figures/fig23_record_by_size_over_time.png" alt="Record by size over time"></td>
<td width="50%"><img src="figures/fig24_small_catches_big.png" alt="Small catches big"></td>
</tr>
</table>

Today's best **4 GB** model (35.3) matches what **30B+** models first reached about **17 months** earlier, with a model ~5× smaller.

### 6️⃣ The decision: what to run on your hardware

<div align="center">
<img src="figures/fig27_payoff_old_rule_vs_new.png" alt="Old rule vs new rule" width="850">
</div>

| Device (4-bit) | Max size | "Biggest that fits" (typical) | ✅ Best that fits | Gain |
|---|---|---|---|---|
| 💻 **4 GB** | ≤ 5.8B | 21.5 | **3Bgeneralv2-ECE-PRYMMAL-Martial** (3.0B), **31.5** | **+10.0** |
| 🖥️ **8 GB** | ≤ 11.6B | 23.2 | **Chocolatine-Fusion-14B** (8.4B), **40.4** | **+17.2** |
| 🧰 **16 GB** | ≤ 23.3B | 38.2 | **T3Q-qwen2.5-14b-v1.0-e3** (15B), **47.1** | **+8.9** |

<table>
<tr>
<td width="50%"><img src="figures/fig26_frontier.png" alt="Efficiency frontier"></td>
<td width="50%"><img src="figures/fig28_top_picks.png" alt="Top picks per device"></td>
</tr>
<tr>
<td align="center"><sub>Efficiency frontier: anything below the orange line is beaten by a smaller model</sub></td>
<td align="center"><sub>Top-8 shortlist per device</sub></td>
</tr>
</table>

### 7️⃣ Size is expensive, and popularity is a poor guide

<table>
<tr>
<td width="50%"><img src="figures/fig29_co2_cost.png" alt="CO2 cost"></td>
<td width="50%"><img src="figures/fig30_popularity_vs_score.png" alt="Popularity vs score"></td>
</tr>
</table>

---

## 🖼️ Full figure gallery

<details>
<summary><b>Part 1 · Meet the data</b> (Figures 1–6)</summary>

| | |
|---|---|
| ![](figures/fig01_size_distribution.png) | ![](figures/fig02_score_distribution.png) |
| ![](figures/fig03_model_types.png) | ![](figures/fig04_families.png) |
| ![](figures/fig05_timeline.png) | ![](figures/fig06_traits.png) |

</details>

<details>
<summary><b>Part 2 · Does size decide capability?</b> (Figures 7–10)</summary>

| | |
|---|---|
| ![](figures/fig07_size_vs_score_main.png) | ![](figures/fig08_density_hexbin.png) |
| ![](figures/fig09_violin_size_groups.png) | ![](figures/fig10_smaller_beats_bigger.png) |

</details>

<details>
<summary><b>Part 3 · Beyond size</b> (Figures 11–17)</summary>

| | |
|---|---|
| ![](figures/fig11_beat_the_curve_histogram.png) | ![](figures/fig12_beat_curve_by_type.png) |
| ![](figures/fig13_beat_curve_by_family.png) | ![](figures/fig14_trait_effects.png) |
| ![](figures/fig15_model_comparison.png) | ![](figures/fig16_feature_importance.png) |
| ![](figures/fig17_predicted_vs_actual.png) | |

</details>

<details>
<summary><b>Part 4 · Which skills come from size?</b> (Figures 18–22)</summary>

| | |
|---|---|
| ![](figures/fig18_correlation_heatmap.png) | ![](figures/fig19_skills_vs_size.png) |
| ![](figures/fig20_skills_ranked_by_size.png) | ![](figures/fig21_radar_profiles.png) |
| ![](figures/fig22_same_score_different_skills.png) | |

</details>

<details>
<summary><b>Part 5 · Are small models catching up?</b> (Figures 23–24)</summary>

| | |
|---|---|
| ![](figures/fig23_record_by_size_over_time.png) | ![](figures/fig24_small_catches_big.png) |

</details>

<details>
<summary><b>Part 6 · The decision</b> (Figures 25–30)</summary>

| | |
|---|---|
| ![](figures/fig25_score_per_gb.png) | ![](figures/fig26_frontier.png) |
| ![](figures/fig27_payoff_old_rule_vs_new.png) | ![](figures/fig28_top_picks.png) |
| ![](figures/fig29_co2_cost.png) | ![](figures/fig30_popularity_vs_score.png) |

</details>

📄 Every figure is explained in detail (*what it shows · how to read it · what the data says · why it matters*) in **[findings_report.md](findings_report.md)**.

---

## 🐍 The code

Everything in this repo is produced by a single Python script, **[`llms_size_vs_capability_analysis.py`](llms_size_vs_capability_analysis.py)** (~2,000 lines), exported from the original **[Google Colab notebook](https://colab.research.google.com/drive/1uwvLyJdW4ZVIPZ5Zl2DY2NIHiNiDj8X2)**. It is split into clearly marked parts, and the `# %%` markers let you run it cell by cell in VS Code, Jupyter or Colab.

| Part | What the code does | Output |
|---|---|---|
| **0 · Setup** | Imports, chart style, and the stated assumptions (`BYTES_PER_PARAM_4BIT = 0.55`, `USABLE_FRACTION = 0.80`, size bands, benchmark names) | — |
| **1 · Load & clean** | Downloads `open-llm-leaderboard/contents` (or reads a local file), renames columns, drops unusable, flagged and duplicate rows, and engineers features | `leaderboard_clean.csv`, Figures 1–6 |
| **2 · Size vs score** | Fits `score ~ log10(params)` with `LinearRegression`, plus binned medians, a hexbin and violins | Figures 7–10 |
| **3 · Beyond size** | Computes `beat_curve` residuals, 95% CIs by type/family/trait, and `RandomForestRegressor` (75/25 split) with permutation importance | Figures 11–17 |
| **4 · Skills** | Spearman correlations, per-benchmark regressions, a radar chart and a same-score/different-skills search | Figures 18–22 |
| **5 · Over time** | Running best score per size band, and how long small models take to catch up with large ones | Figures 23–24 |
| **6 · Decision** | Score per GB, efficiency frontier, old vs new rule, top-8 picks per device, CO₂ and popularity | Figures 25–30, `interactive_explorer.html` |
| **7–8 · Report** | Writes the conclusions and `findings_report.md`, then zips everything | `llm_analysis_outputs.zip` |

### ▶️ Run it yourself

**Locally**

```bash
pip install -r requirements.txt
python llms_size_vs_capability_analysis.py
```

**On Google Colab (easiest):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uwvLyJdW4ZVIPZ5Zl2DY2NIHiNiDj8X2)

Open the [original notebook](https://colab.research.google.com/drive/1uwvLyJdW4ZVIPZ5Zl2DY2NIHiNiDj8X2) and choose **Runtime → Run all**. Alternatively, upload the `.py` file to any Colab notebook and run `%run llms_size_vs_capability_analysis.py` in a cell. The results zip downloads automatically.

The data (~1 MB) downloads automatically from Hugging Face, and a full run takes about 1–2 minutes. All outputs go to `llm_analysis_outputs/`. Optional environment variables:

| Variable | Effect |
|---|---|
| `LEADERBOARD_FILE=path/to/file.parquet` | Use a local copy of the raw leaderboard instead of downloading it |
| `STRICT=1` | Stop on the first chart error (useful for testing) |

Want different assumptions? Edit `BYTES_PER_PARAM_4BIT`, `USABLE_FRACTION`, `BUDGETS_GB` or `BAND_EDGES` at the top of Part 0 and re-run. Every figure and number updates.

---

## 🖱️ Interactive explorer

[`interactive_explorer.html`](interactive_explorer.html) is an interactive Plotly version of the main chart. Hover over any dot to see the model's name, size, memory need and beat-the-curve score. Click legend entries to filter, and drag to zoom.

▶️ **[Open the live explorer](https://htmlpreview.github.io/?https://github.com/06pratyush/llms_size_vs_capability/blob/main/interactive_explorer.html)**, or download the file and open it in any browser.

---

## 📂 Repository structure

```
llms_size_vs_capability/
├── llms_size_vs_capability_analysis.py  # Full pipeline (exported from the Colab notebook)
├── requirements.txt           # Python dependencies
├── leaderboard_clean.csv      # 4,486 cleaned + enriched models (52 columns)
├── findings_report.md         # Full written analysis, all 30 figures explained
├── interactive_explorer.html  # Plotly interactive size-vs-score chart
├── figures/                   # 30 publication-quality PNG charts
│   ├── fig01_size_distribution.png
│   ├── ...
│   └── fig30_popularity_vs_score.png
├── LICENSE                    # MIT license
├── NOTICE                     # Third-party data attribution
└── README.md
```

---

## 📑 Data dictionary: `leaderboard_clean.csv`

<details>
<summary><b>Click to expand all 52 columns</b></summary>

| Group | Column | Description |
|---|---|---|
| **Identity** | `eval_name` | Unique evaluation id (model + precision) |
| | `model_id` | Hugging Face repo id, e.g. `ibm-granite/granite-3.0-1b-a400m-base` |
| | `Model sha` | Exact commit evaluated |
| | `architecture` | Transformers class, e.g. `LlamaForCausalLM` |
| | `family`, `family_group` | Blueprint family (Llama, Qwen, Mistral, …) and grouped version |
| | `base_model` | Model it was derived from |
| **How it was made** | `model_type_raw`, `T`, `model_type` | Raw leaderboard type, emoji code, cleaned type (Pretrained / Fine-tuned / Chat-tuned / Merge / Other) |
| | `weight_type`, `is_adapter` | Original weights vs adapter/delta |
| | `is_merged`, `is_official`, `is_moe` | Merge flag, official lab release, mixture-of-experts |
| | `has_chat_template`, `generation` | Chat-ready; number of training generations on top of a base |
| **Size & memory** | `params_b`, `log_params` | Parameters in billions, and log10 of that |
| | `precision`, `precision_group` | Evaluated precision (bfloat16, float16, …) |
| | `footprint_4bit_gb`, `footprint_16bit_gb` | Estimated memory at 4-bit and 16-bit |
| | `fits_4gb`, `fits_8gb`, `fits_16gb` | Fits that device at 4-bit using 80% of memory |
| | `size_band` | Tiny / Small / Medium / Large / X-Large / Huge |
| **Scores** | `avg_score` | Leaderboard average (0 = random, 100 = perfect) |
| | `ifeval`, `bbh`, `math_lvl5`, `gpqa`, `musr`, `mmlu_pro` | Normalised benchmark scores |
| | `IFEval Raw`, `BBH Raw`, … | Raw accuracy before normalisation |
| **Derived** | `expected_score` | Score predicted from size alone |
| | `beat_curve` | `avg_score − expected_score` (over/under-performance for its size) |
| | `score_per_gb` | Score per GB of 4-bit memory |
| **Meta** | `license`, `hub_likes`, `on_hub`, `is_flagged` | Model licence, likes, still downloadable, flagged |
| | `co2_kg` | Estimated CO₂ for the evaluation run |
| | `upload_date`, `submission_date`, `model_date`, `years_since_2023` | Dates, and time since Jan 2023 |

</details>

### Quick start

```python
import pandas as pd

df = pd.read_csv("leaderboard_clean.csv")

# Best downloadable models that fit an 8 GB device
picks = (df[df.on_hub & df.fits_8gb]
         .nlargest(10, "avg_score")[["model_id", "params_b", "avg_score", "beat_curve"]])
print(picks)
```

---

## ⚠️ Limitations

- **Frozen snapshot.** The leaderboard was retired; the last evaluation is from March 2025, so newer models are missing.
- **Academic benchmarks** are not your task. Always test the shortlist on your real workload.
- **Benchmark tuning.** Some community models, especially merges, may be tuned to these exact tests. Official releases are the safer choice.
- **Memory estimate.** 0.55 GB per billion parameters at 4-bit; longer contexts need more.
- **Correlation, not causation.** Blueprint, date and training type are tangled together.

---

## 🙏 Acknowledgements

- Data: [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) by Hugging Face, via the [`open-llm-leaderboard/contents`](https://huggingface.co/datasets/open-llm-leaderboard/contents) dataset.
- Benchmarks: IFEval, BBH, MATH, GPQA, MuSR, MMLU-PRO, each credited to its original authors.

## 📜 License

The analysis, derived data, figures, report and explorer are released under the **[MIT License](LICENSE)** (see [NOTICE](NOTICE) for data attribution). The upstream leaderboard data belongs to its original publishers, and each model listed is governed by its own licence (see the `license` column).

<div align="center">

⭐ If this helped you pick a model, consider starring the repo!

</div>
