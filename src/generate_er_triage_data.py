import pandas as pd
import numpy as np
# from scipy.stats import truncnorm

# --- Configuration ---
NUM_PATIENTS = 15000
VITAL_SIGNS = [
    "Heart_Rate_BPM",
    "Respiratory_Rate_BPM",
    "Systolic_BP_mmHg",
    "Diastolic_BP_mmHg",
    "SpO2_Percent",
    "Temperature_C",
    "HRV_ms",
    "Mood_Score_1_5" # 1=Very Bad, 5=Very Good
]

# --- Risk Criteria (Simplified ESI/NEWS-like) ---
# A patient is High Risk (1) if any of the following are true:
# 1. HR < 40 or HR > 130
# 2. RR < 8 or RR > 25
# 3. SBP < 90 or SBP > 200
# 4. SpO2 < 90%
# 5. Temp < 35.0 or Temp > 39.0
# 6. Mood Score is 1 (Very Bad) AND another vital sign is abnormal

# --- Data Generation Functions ---

def get_truncated_normal(mean, sd, low, high):
    """Generates a truncated normal distribution using a simpler rejection sampling method."""
    while True:
        x = np.random.normal(mean, sd)
        if low <= x <= high:
            return x

def get_truncated_normal_rvs(mean, sd, low, high, size=1):
    """Generates an array of truncated normal samples."""
    results = []
    for _ in range(size):
        results.append(get_truncated_normal(mean, sd, low, high))
    return np.array(results)
    """Generates a truncated normal distribution."""
    return truncnorm(
        (low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd
    )

def generate_vitals(is_high_risk):
    """Generates a single patient's vital signs based on risk level."""
    vitals = {}

    # Base distributions (Normal/Low Risk)
    # Base distributions (Normal/Low Risk)
    # Define parameters for a simple normal distribution approximation
    hr_mean, hr_sd, hr_low, hr_high = 75, 15, 40, 130
    rr_mean, rr_sd, rr_low, rr_high = 16, 3, 8, 25
    sbp_mean, sbp_sd, sbp_low, sbp_high = 120, 15, 90, 200
    dbp_mean, dbp_sd, dbp_low, dbp_high = 80, 10, 50, 120
    spo2_mean, spo2_sd, spo2_low, spo2_high = 97, 1.5, 90, 100
    temp_mean, temp_sd, temp_low, temp_high = 37.0, 0.5, 35.0, 39.0
    hrv_mean, hrv_sd, hrv_low, hrv_high = 50, 15, 10, 100
    mood_mean, mood_sd, mood_low, mood_high = 4, 1, 1, 5

    vitals["Heart_Rate_BPM"] = int(get_truncated_normal_rvs(hr_mean, hr_sd, hr_low, hr_high)[0])
    vitals["Respiratory_Rate_BPM"] = int(get_truncated_normal_rvs(rr_mean, rr_sd, rr_low, rr_high)[0])
    vitals["Systolic_BP_mmHg"] = int(get_truncated_normal_rvs(sbp_mean, sbp_sd, sbp_low, sbp_high)[0])
    vitals["Diastolic_BP_mmHg"] = int(get_truncated_normal_rvs(dbp_mean, dbp_sd, dbp_low, dbp_high)[0])
    vitals["SpO2_Percent"] = round(get_truncated_normal_rvs(spo2_mean, spo2_sd, spo2_low, spo2_high)[0], 1)
    vitals["Temperature_C"] = round(get_truncated_normal_rvs(temp_mean, temp_sd, temp_low, temp_high)[0], 1)
    vitals["HRV_ms"] = int(get_truncated_normal_rvs(hrv_mean, hrv_sd, hrv_low, hrv_high)[0])
    vitals["Mood_Score_1_5"] = int(round(get_truncated_normal_rvs(mood_mean, mood_sd, mood_low, mood_high)[0]))

    if is_high_risk:
        # Introduce an abnormality for high-risk patients
        abnormality_type = np.random.choice([
            "HR_Abnormal", "RR_Abnormal", "SBP_Abnormal", "SpO2_Abnormal",
            "Temp_Abnormal", "Mood_Abnormal_Combined"
        ])

        if abnormality_type == "HR_Abnormal":
            if np.random.rand() < 0.5: # Tachycardia
                vitals["Heart_Rate_BPM"] = int(get_truncated_normal_rvs(140, 10, 131, 180)[0])
            else: # Bradycardia
                vitals["Heart_Rate_BPM"] = int(get_truncated_normal_rvs(35, 5, 20, 39)[0])

        elif abnormality_type == "RR_Abnormal":
            if np.random.rand() < 0.5: # Tachypnea
                vitals["Respiratory_Rate_BPM"] = int(get_truncated_normal_rvs(30, 5, 26, 40)[0])
            else: # Bradypnea
                vitals["Respiratory_Rate_BPM"] = int(get_truncated_normal_rvs(6, 1, 4, 7)[0])

        elif abnormality_type == "SBP_Abnormal":
            if np.random.rand() < 0.7: # Hypotension (more common in critical)
                vitals["Systolic_BP_mmHg"] = int(get_truncated_normal_rvs(80, 5, 60, 89)[0])
            else: # Severe Hypertension
                vitals["Systolic_BP_mmHg"] = int(get_truncated_normal_rvs(210, 10, 201, 250)[0])
            # Adjust DBP to maintain a plausible pulse pressure
            vitals["Diastolic_BP_mmHg"] = int(vitals["Systolic_BP_mmHg"] * np.random.uniform(0.6, 0.75))

        elif abnormality_type == "SpO2_Abnormal":
            vitals["SpO2_Percent"] = round(get_truncated_normal_rvs(85, 3, 70, 89.9)[0], 1)

        elif abnormality_type == "Temp_Abnormal":
            if np.random.rand() < 0.5: # Fever
                vitals["Temperature_C"] = round(get_truncated_normal_rvs(39.5, 0.5, 39.1, 41.0)[0], 1)
            else: # Hypothermia
                vitals["Temperature_C"] = round(get_truncated_normal_rvs(34.0, 0.5, 32.0, 34.9)[0], 1)

        elif abnormality_type == "Mood_Abnormal_Combined":
            vitals["Mood_Score_1_5"] = 1 # Very Bad Mood
            # Combine with a less severe but still abnormal vital sign
            vitals["HRV_ms"] = int(get_truncated_normal_rvs(15, 5, 5, 25)[0]) # Very low HRV

    return vitals

def assign_risk_label(vitals):
    """Applies the risk criteria to assign the final label (0=Low, 1=High)."""
    hr = vitals["Heart_Rate_BPM"]
    rr = vitals["Respiratory_Rate_BPM"]
    sbp = vitals["Systolic_BP_mmHg"]
    spo2 = vitals["SpO2_Percent"]
    temp = vitals["Temperature_C"]
    mood = vitals["Mood_Score_1_5"]

    # Check for primary high-risk criteria
    if hr < 40 or hr > 130: return 1
    if rr < 8 or rr > 25: return 1
    if sbp < 90 or sbp > 200: return 1
    if spo2 < 90.0: return 1
    if temp < 35.0 or temp > 39.0: return 1

    # Check for secondary high-risk criteria (Mood combined with other abnormality)
    is_abnormal_secondary = (hr < 50 or hr > 110) or \
                            (rr < 10 or rr > 20) or \
                            (sbp < 100 or sbp > 160) or \
                            (spo2 < 95.0) or \
                            (temp < 36.0 or temp > 38.0) or \
                            (vitals["HRV_ms"] < 20)

    if mood == 1 and is_abnormal_secondary:
        return 1

    return 0

# --- Main Generation Logic ---

# Target ratio: 80% Low Risk (0), 20% High Risk (1)
# This is a common split for triage datasets, ensuring enough positive cases for training.
num_high_risk = int(NUM_PATIENTS * 0.20)
num_low_risk = NUM_PATIENTS - num_high_risk

data = []

# Generate High Risk patients
for _ in range(num_high_risk):
    vitals = generate_vitals(is_high_risk=True)
    vitals["Risk_Label"] = assign_risk_label(vitals)
    data.append(vitals)

# Generate Low Risk patients
for _ in range(num_low_risk):
    vitals = generate_vitals(is_high_risk=False)
    vitals["Risk_Label"] = assign_risk_label(vitals)
    data.append(vitals)

# Create DataFrame and shuffle
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)

# Ensure the final dataset has the correct number of high-risk patients
# The generation logic is designed to ensure the 'High Risk' generation path
# results in a 'Risk_Label' of 1, but a final check is important.
final_high_risk_count = df['Risk_Label'].sum()
print(f"Total Patients: {NUM_PATIENTS}")
print(f"Generated High Risk Count: {final_high_risk_count}")
print(f"Generated Low Risk Count: {NUM_PATIENTS - final_high_risk_count}")

# Save to CSV
df.to_csv("er_triage_dataset_15k.csv", index=False)
print("Dataset saved to er_triage_dataset_15k.csv")
