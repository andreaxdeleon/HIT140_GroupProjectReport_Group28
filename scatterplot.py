import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data based on your output (replace with your actual data)
data = {
    'Actual': [5, 3, 4, 3, 2],
    'Predicted': [3.157823, 3.332667, 3.114112, 3.201534, 3.332667]
}

# Create a DataFrame
results_df = pd.DataFrame(data)

# Create a scatter plot for actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Actual', y='Predicted', color='blue', label='Data Points')
plt.plot(results_df['Actual'], results_df['Actual'], color='red', linestyle='--', label='Perfect Prediction')  # Diagonal line for perfect predictions
plt.title('Actual vs. Predicted Well-being Scores')
plt.xlabel('Actual Well-being Score')
plt.ylabel('Predicted Well-being Score')
plt.xlim(1, 6)  # Adjust based on the range of your data
plt.ylim(1, 6)  # Adjust based on the range of your data
plt.axhline(y=0, color='black', linewidth=0.8)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.legend()
plt.grid()
plt.show()
