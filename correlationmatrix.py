import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

# Merge datasets on a common column (e.g., 'ID')
merged_df = pd.merge(df1, df2, on='ID')
merged_df = pd.merge(merged_df, df3, on='ID')

# Display the first few rows of the dataset
print("Dataset Head:")
print(merged_df.head())

# Specify the column to transform (replace 'C_we' with the actual column name you want to log transform)
column_to_transform = 'C_we'  # Example column name

# Check for zero or negative values before applying log transformation
if (merged_df[column_to_transform] <= 0).any():
    print("Warning: Log transformation cannot be applied to non-positive values.")
else:
    # Apply log transformation
    merged_df[column_to_transform + '_log'] = np.log(merged_df[column_to_transform])
    print(f"Log transformation applied to {column_to_transform}. New column: {column_to_transform + '_log'}")

# Calculate the correlation matrix
correlation_matrix = merged_df.corr()

# Plotting the correlation matrix with adjusted font sizes
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
            cbar_kws={"shrink": .8}, annot_kws={"size": 7})  # Adjust "size" for font size in annotations
plt.title('Correlation Matrix', fontsize=16)  # Adjust "fontsize" for the title
plt.xticks(fontsize=10)  # Adjust "fontsize" for x-axis labels
plt.yticks(fontsize=10)  # Adjust "fontsize" for y-axis labels
plt.show()