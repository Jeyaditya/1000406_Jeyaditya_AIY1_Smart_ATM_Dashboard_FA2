## 1000406_Jeyaditya_AIY1_Smart_ATM_Dashboard_FA2
# ATM Intelligence Demand Forecasting
### FinTrust Bank Ltd. | Data Mining Project (FA-1 & FA-2)

## Project Overview
This project, developed for **FinTrust Bank Ltd.**, addresses the challenge of balancing cash across a network of over 3,000 ATMs. Using advanced data mining techniques, the application identifies usage trends, clusters ATMs by demand behavior, and detects anomalies to prevent cash-out events during festivals and salary dates.



## Core Features:
The application is structured into three main stages following the **WACP Data Mining rubric**:

**Exploratory Data Analysis (EDA):** 
    * Visual storytelling to uncover trends and relationships between variables like weather, holidays, and withdrawals.
**K-Means Clustering:** 
    * Segmenting ATMs into "High-Demand," "Steady-Demand," and "Low-Demand" groups based on behavior.
**Anomaly Detection:** 
    * Utilizing **Isolation Forest** algorithms to highlight unusual spikes in cash demand during special events or holidays.
**Interactive Python Script:** 
    * A reproducible Streamlit dashboard allowing users to filter by features like `Day_of_Week` and `Location_Type`.

## Preprocessing Logic:
Before analysis, the raw dataset undergoes a robust preprocessing pipeline established in FA-1:
**Handling Missing Values:** 
    * `Holiday_Flag` blanks are filled with `0` (No Holiday) to maintain binary consistency.
**Date Transformation:** 
    * Raw dates are converted to datetime objects to extract `Month`, `Week Number`, and `Day of Week`.
**Categorical Encoding:** 
    * Textual categories are mapped to numbers (e.g., Urban=1, Semi-Urban=2, Rural=3).
**Normalization:** 
    * Min-Max scaling is applied to ensure features like `Withdrawals` do not dominate the model due to scale disparity.


## Mathematical Derivations
To ensure technical accuracy, the following formulas are integrated into the logic:

1.  **Standardization ($z$-score):** $$z = \frac{x - \mu}{\sigma}$$ 
    * Used to center data before clustering, ensuring features with different units are comparable.
2.  **Euclidean Distance:** $$d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$ 
        * The core metric for **K-Means Clustering** to determine the proximity of an ATM's behavior to a cluster centroid.
3.  **Anomaly Score (Isolation Forest):** $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$ 
    Calculates the isolation of a data point; the shorter the path length $h(x)$, the more likely the point is an anomaly.

## Installation & Usage

### 1. Requirements
Ensure you have **Python 3.8+** installed. Download the required libraries:
 put this in the windows powershell:
pip install streamlit pandas numpy plotly scikit-learn Pillow

### 2. File Structure
* ATM_app.py: The interactive Streamlit dashboard.
* atm_cash_management_dataset.csv: The processed source data.
* logo.png: Professional branding icon.
* README.md: Project documentation.

### 3. Running the App
Navigate to the project directory and run:
    streamlit run ATM_app.py

## Author Information

 * Name: A Jeyaditya (JD)
 * Registration Number: 1000406
 * CRS Facillitator: Aruljothi
 * School: Jain Vidyalaya IB World Schoo
