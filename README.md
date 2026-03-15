# 1000406_Jeyaditya_AIY1_Smart_ATM_Dashboard_FA2
# 🏦 ATM Intelligence Demand Forecasting
### FinTrust Bank Ltd. | Data Mining Project (FA-1 & FA-2)

## Project Overview
[cite_start]This project, developed for **FinTrust Bank Ltd.**, addresses the challenge of balancing cash across a network of over 3,000 ATMs[cite: 248]. [cite_start]Using advanced data mining techniques, the application identifies usage trends, clusters ATMs by demand behavior, and detects anomalies to prevent cash-out events during festivals and salary dates[cite: 125, 248].



## Core Features (FA-2)
[cite_start]The application is structured into three main stages following the **WACP Data Mining rubric**[cite: 125]:

* [cite_start]**Exploratory Data Analysis (EDA):** Visual storytelling to uncover trends and relationships between variables like weather, holidays, and withdrawals[cite: 125, 148, 168].
* [cite_start]**K-Means Clustering:** Segmenting ATMs into "High-Demand," "Steady-Demand," and "Low-Demand" groups based on behavior[cite: 125, 176, 183, 191].
* [cite_start]**Anomaly Detection:** Utilizing **Isolation Forest** algorithms to highlight unusual spikes in cash demand during special events or holidays[cite: 125, 199, 203].
* [cite_start]**Interactive Python Script:** A reproducible Streamlit dashboard allowing users to filter by features like `Day_of_Week` and `Location_Type`[cite: 125, 215, 221].

## Preprocessing Logic (FA-1)
[cite_start]Before analysis, the raw dataset undergoes a robust preprocessing pipeline established in FA-1[cite: 248]:
* [cite_start]**Handling Missing Values:** `Holiday_Flag` blanks are filled with `0` (No Holiday) to maintain binary consistency[cite: 279, 335].
* [cite_start]**Date Transformation:** Raw dates are converted to datetime objects to extract `Month`, `Week Number`, and `Day of Week`[cite: 287, 292, 335].
* [cite_start]**Categorical Encoding:** Textual categories are mapped to numbers (e.g., Urban=1, Semi-Urban=2, Rural=3)[cite: 295, 307, 335].
* [cite_start]**Normalization:** Min-Max scaling is applied to ensure features like `Withdrawals` do not dominate the model due to scale disparity[cite: 308, 317, 335].



## Mathematical Derivations
To ensure technical accuracy, the following formulas are integrated into the logic:

1.  **Standardization ($z$-score):** $$z = \frac{x - \mu}{\sigma}$$ 
    [cite_start]Used to center data before clustering, ensuring features with different units are comparable[cite: 317].
2.  **Euclidean Distance:** $$d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$ 
    [cite_start]The core metric for **K-Means Clustering** to determine the proximity of an ATM's behavior to a cluster centroid[cite: 178].
3.  **Anomaly Score (Isolation Forest):** $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$ 
    Calculates the isolation of a data point; the shorter the path length $h(x)$, the more likely the point is an anomaly[cite: 203].

## 🚀 Installation & Usage

### 1. Requirements
Ensure you have **Python 3.8+** installed. Download the required libraries:
```bash
pip install streamlit pandas numpy plotly scikit-learn Pillow
