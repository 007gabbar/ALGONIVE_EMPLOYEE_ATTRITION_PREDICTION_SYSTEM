# ALGONIVE_EMPLOYEE-ATTRITION-PREDICTION-SYSTEM

## 📌 Project Overview
This machine learning project predicts whether an employee is likely to leave a company based on various HR metrics. By proactively identifying at-risk employees, human resources teams can take targeted actions to improve retention, saving significant recruitment and training costs.

## 📊 The Dataset
The project utilizes the **IBM HR Analytics Employee Attrition & Performance** dataset (`IBM_Employee_Attrition.csv`). 
* **Size:** 1,470 employee records.
* **Features:** 35 attributes including Demographics (Age, Gender), Role specifics (Department, Job Role), and Satisfaction metrics (Job Satisfaction, Work-Life Balance).

## 🛠️ Technologies Used
* **Python 3.x**
* **Pandas & NumPy:** Data manipulation and preprocessing.
* **Scikit-Learn:** Machine learning modeling (Random Forest Classifier).
* **Matplotlib & Seaborn:** Data visualization.

## 🚀 Model Performance
A Random Forest Classifier was trained using class balancing to account for the natural skew in retention vs. attrition data.
* **Overall Accuracy:** 83.67%
* **Key Finding:** The model is highly accurate at identifying retained employees (98% recall).

## 📈 Key Insights (Feature Importance)
The model identified the top three drivers of employee turnover in this dataset:
1. **Monthly Income:** Lower income levels strongly correlate with higher attrition risk.
2. **Age:** Younger employees show a higher tendency to leave compared to older staff.
3. **Daily Rate:** Fluctuations in daily compensation rates impact retention.

*(View `feature_importance.png` and `confusion_matrix.png` in the repository for visual breakdowns).*

## 💻 How to Run Locally
1. Clone the repository:
   `git clone https://github.com/your-username/HR-Attrition-Prediction-System.git`
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Run the model script:
   `python attrition_model.py`
