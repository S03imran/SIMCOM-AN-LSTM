import pandas as pd
import matplotlib.pyplot as plt

# Read data from the Excel file
data_file = 'paper_results.xls'
df = pd.read_excel(data_file)

# Ensure the columns are correctly named (adjust if necessary)
# Assuming the Excel file has columns named 'ratio', 'lstm', 'cn', 'aa', 'jc', 'pa', 'nlc', 'cclp', 'cosp', 'sp', 'car', 'lp', 'l3', 'act', 'mfi'
features = ['SimCom-AN-LSTM', 'CN', 'AA', 'JC', 'PA', 'NLC', 'CCLP', 'COSP', 'SP', 'CAR', 'LP', 'L3', 'ACT', 'MFI']

# Plotting the results
plt.figure(figsize=(10, 6))

for feature in features:
    values = df[feature]
    plt.plot(df['Ratio'], values, marker='o', label=feature)

plt.xlabel('Ratio')
plt.ylabel('AUC Value')
plt.title('AUC Values for Different Features on eu-core Dataset')
plt.legend()
plt.grid(True)
plt.show()
