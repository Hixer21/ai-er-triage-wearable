# AI-Powered ER Triage with Low-Cost Wearables

This project aims to develop a machine learning model to quickly and accurately stratify Emergency Room (ER) patients into **Low-Risk** and **High-Risk** categories using vital signs collected from a low-cost wearable device (mimicking the VitalWatch). The goal is to reduce the burden on ER staff by identifying low-risk patients early, improving overall efficiency and patient flow.

## Project Structure

*   `data/`: Contains the synthetic patient dataset (`er_triage_dataset_15k.csv`) used for model training.
*   `src/`: Contains all Python scripts for data generation, analysis, and model training.
    *   `train_model.py`: The main script for training and evaluating the AI triage model.
    *   `generate_er_triage_data.py`: Script used to create the synthetic dataset.
    *   `analyze_triage_data.py`: Script used for initial data validation and analysis.
*   `docs/`: Contains the initial research report and documentation.
    *   `AI_ER_Triage_Research_Report.md`: Initial research on the VitalWatch protocol and clinical risk criteria.
*   `models/`: Directory to store trained model files.

## Dataset

The synthetic dataset contains 15,000 patient records with the following features:

| Feature | Description |
| :--- | :--- |
| `Heart_Rate_BPM` | Heart Rate in Beats per Minute |
| `Respiratory_Rate_BPM` | Respiratory Rate in Breaths per Minute (Simulated) |
| `Systolic_BP_mmHg` | Systolic Blood Pressure in mmHg |
| `Diastolic_BP_mmHg` | Diastolic Blood Pressure in mmHg |
| `SpO2_Percent` | Peripheral Oxygen Saturation in % |
| `Temperature_C` | Body Temperature in Celsius |
| `HRV_ms` | Heart Rate Variability in milliseconds |
| `Mood_Score_1_5` | Subjective mood score (1=Very Bad, 5=Very Good) |
| `Risk_Label` | **Target Variable:** 0 (Low Risk) or 1 (High Risk) |

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hixer21/ai-er-triage-wearable.git
    cd ai-er-triage-wearable
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Train the model:**
    ```bash
    python src/train_model.py
    ```
