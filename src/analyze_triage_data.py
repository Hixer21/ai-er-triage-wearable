import pandas as pd

# Load the generated dataset
import io
output_buffer = io.StringIO()
df = pd.read_csv("er_triage_dataset_15k.csv")

# 1. Basic Data Validation
print("--- Dataset Head and Info ---")
print(df.head())
print("\n")
df.info(buf=output_buffer)
print(output_buffer.getvalue())
print("\n")

# 2. Risk Distribution Check
risk_counts = df['Risk_Label'].value_counts()
risk_percentage = df['Risk_Label'].value_counts(normalize=True) * 100
print("--- Risk Label Distribution ---")
print(f"Total Patients: {len(df)}")
print(f"Low Risk (0) Count: {risk_counts.get(0, 0)}")
print(f"High Risk (1) Count: {risk_counts.get(1, 0)}")
print(f"Low Risk (0) Percentage: {risk_percentage.get(0, 0):.2f}%")
print(f"High Risk (1) Percentage: {risk_percentage.get(1, 0):.2f}%")
print("\n")

# 3. Descriptive Statistics for Low Risk vs High Risk
print("--- Descriptive Statistics for Low Risk Patients (Risk_Label = 0) ---")
print(df[df['Risk_Label'] == 0].describe())
print("\n")

print("--- Descriptive Statistics for High Risk Patients (Risk_Label = 1) ---")
print(df[df['Risk_Label'] == 1].describe())
print("\n")

# 4. Check for Outliers/Abnormalities (e.g., min/max for High Risk)
print("--- High Risk Patient Vitals Check (Min/Max) ---")
high_risk_df = df[df['Risk_Label'] == 1]
for col in ['Heart_Rate_BPM', 'Respiratory_Rate_BPM', 'Systolic_BP_mmHg', 'SpO2_Percent', 'Temperature_C']:
    print(f"{col}: Min={high_risk_df[col].min()}, Max={high_risk_df[col].max()}")

# Save the analysis results to a file
with open("triage_data_analysis_report.txt", "w") as f:
    f.write("--- Dataset Head and Info ---\n")
    f.write(df.head().to_string() + "\n\n")
    df.info(buf=output_buffer)
    f.write(output_buffer.getvalue() + "\n")
    f.write("--- Risk Label Distribution ---\n")
    f.write(f"Total Patients: {len(df)}\n")
    f.write(f"Low Risk (0) Count: {risk_counts.get(0, 0)}\n")
    f.write(f"High Risk (1) Count: {risk_counts.get(1, 0)}\n")
    f.write(f"Low Risk (0) Percentage: {risk_percentage.get(0, 0):.2f}%\n")
    f.write(f"High Risk (1) Percentage: {risk_percentage.get(1, 0):.2f}%\n\n")
    f.write("--- Descriptive Statistics for Low Risk Patients (Risk_Label = 0) ---\n")
    f.write(df[df['Risk_Label'] == 0].describe().to_string() + "\n\n")
    f.write("--- Descriptive Statistics for High Risk Patients (Risk_Label = 1) ---\n")
    f.write(df[df['Risk_Label'] == 1].describe().to_string() + "\n\n")
    f.write("--- High Risk Patient Vitals Check (Min/Max) ---\n")
    for col in ['Heart_Rate_BPM', 'Respiratory_Rate_BPM', 'Systolic_BP_mmHg', 'SpO2_Percent', 'Temperature_C']:
        f.write(f"{col}: Min={high_risk_df[col].min()}, Max={high_risk_df[col].max()}\n")

print("Analysis report saved to triage_data_analysis_report.txt")
