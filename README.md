# Style-Aware LLM Ranking  
### Controlling for Response Length and Formatting Bias in Human Preference Leaderboards

## Overview
Human preference leaderboards are commonly used to compare large language models (LLMs). However, these rankings often conflate **intrinsic model capability** with **stylistic presentation effects**, such as longer responses or structured formatting.

This project demonstrates that **response length alone has a large and statistically significant impact on model rankings**, and introduces a **style-aware Bradley–Terry ranking framework** that controls for such biases.

By explicitly modeling stylistic covariates alongside model identity, we produce a **length-controlled leaderboard** that better reflects true model performance.

---

## Key Questions
- Do stylistic factors systematically bias human preference judgments?
- How large is the effect of response length compared to model identity?
- How do rankings change once stylistic effects are controlled?

---

## Dataset Overview and Coverage

We analyze a large-scale human preference dataset consisting of pairwise battles between LLMs. Each battle contains:
- Two model responses to the same prompt
- A human-labeled outcome (win / loss / tie)
- Full conversation transcripts
- Rich metadata (token counts, formatting statistics, language, turns)

### Battle Participation
![Battle Count per Model](results/figures/newplot%20(11).png)

### Outcome Distribution
![Outcome Distribution](results/figures/newplot%20(12).png)

These plots show that:
- No single model dominates participation
- Wins and losses are well balanced
- Rankings are not driven by sparse data artifacts

---

## Pairwise Coverage and Reliability

### Pairwise Battle Counts
![Pairwise Battle Counts](results/figures/newplot%20(13).png)

### Pairwise Battle Counts (No Ties)
![Pairwise No-Tie Counts](results/figures/newplot%20(14).png)

The dense pairwise structure ensures statistically meaningful comparisons between models.

---

## Baseline Ranking (No Style Control)

We first construct a standard leaderboard using a **Bradley–Terry logistic regression model**, where each battle contributes a pairwise preference signal.

### Model Scores with Confidence Intervals
![Model Scores CI](results/figures/newplot%20(22).png)

### Baseline Rank Heatmap
![Baseline Rank Heatmap](results/figures/newplot%20(23).png)

These rankings reflect raw human preferences without accounting for stylistic bias.

---

## Evidence of Length Bias

To test whether verbosity influences rankings, we analyze response length across models.

### Average Response Length vs Rank
![Length vs Rank](results/figures/newplot%20(24).png)

**Key observation:**  
Higher-ranked models tend to produce longer responses, indicating that verbosity alone can inflate preference scores.

---

## Style Feature Modeling

We introduce explicit stylistic covariates into the ranking model:
- Total assistant token count (length)
- List usage
- Header usage
- Bold formatting

Each feature is encoded as a **normalized difference** between competing responses.

### Style Feature Coefficients
![Style Feature Coefficients](results/figures/newplot%20(30).png)

**Result:**  
Response length has a stronger effect than most model identity coefficients, while formatting features contribute comparatively little.

---

## Style-Controlled Ranking

We augment the Bradley–Terry model to jointly learn:
- Intrinsic model strength
- Independent stylistic effects

### Ranking Comparison (With vs Without Style Control)
![Ranking Comparison](results/figures/newplot%20(25).png)

### Rank Heatmap with Style Control
![Style-Controlled Heatmap](results/figures/newplot%20(26).png)

**Interpretation:**
- Some models drop after length control, revealing style inflation
- Others rise, indicating stronger intrinsic performance
- These effects are invisible in standard leaderboards

---

## Per-Category Performance

Models are also evaluated across task categories.

### Overall and Per-Category Rankings
![Category Rankings](results/figures/newplot%20(21).png)

This shows that:
- Model strengths vary by task type
- Style bias affects categories differently

---

## Additional Diagnostics and Supporting Analysis

These plots provide additional context and validation:

- **Language Distribution**  
  ![Languages](results/figures/newplot%20(15).png)

- **Conversation Turn Counts**  
  ![Turns](results/figures/newplot%20(16).png)

- **Win-Rate Matrix (A vs B)**  
  ![Win Matrix](results/figures/newplot%20(17).png)

- **Average Win Rate (All Models)**  
  ![Avg Win Rate](results/figures/newplot%20(18).png)

- **Win Rate Excluding Top Prompts**  
  ![No Top Prompts](results/figures/newplot%20(19).png)

- **Category Distribution**  
  ![Category Dist](results/figures/newplot%20(20).png)

---

## Prompt Clustering Analysis

To analyze prompt diversity and structure, we cluster prompts using dimensionality reduction.

- **Runtime vs Clusters (K)**  
  ![Runtime](results/figures/newplot%20(27).png)

- **Elbow Plot**  
  ![Elbow](results/figures/newplot%20(28).png)

- **2D Prompt Visualization**  
  ![Prompt Clusters](results/figures/newplot%20(29).png)

---

## Key Contributions
- Quantified response-length bias in human LLM evaluations
- Built a style-aware Bradley–Terry ranking framework
- Introduced bootstrapped confidence intervals for ranking stability
- Demonstrated ranking shifts after controlling for style
- Delivered a reproducible, research-grade evaluation pipeline

---

## Why This Matters
Leaderboards influence deployment, funding, and research direction.  
This project shows that **presentation can masquerade as capability**, and provides a principled method to correct for it.

---

## Next Steps
- Extend style controls (tone, safety verbosity, hedging)
- Apply framework to new evaluation datasets
- Explore causal interpretations of stylistic bias
