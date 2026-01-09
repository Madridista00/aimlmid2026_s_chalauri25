# aimlmid2026_s_chalauri25
Task 1 – Finding the Correlation

Objective: Find Pearson's correlation coefficient for the given dataset and create a scatter plot with the regression line.

Code: task1_correlation.py

import numpy as np
import matplotlib.pyplot as plt

x = np.array([-9, -7, -5, -3.5, -1.0, 1, 3, 5, 7.9, 9.9])
y = np.array([4, 4.5, 3, 4, 1.0, 1.3, -2, -3.6, -4.9, -5.8])

# Pearson's correlation coefficient
r = np.corrcoef(x, y)[0, 1]
print("Pearson's r:", r)

# Line of best fit
slope, intercept = np.polyfit(x, y, 1)

# Scatter plot
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, slope * x + intercept, color='red', label='Line of Best Fit')
plt.title('Scatter Plot of Data Points with Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.savefig('correlation_plot.png')
plt.show()


Result:

Pearson's correlation coefficient (r): ~ -0.968

The plot shows a strong negative correlation between X and Y. The red line represents the linear regression fit.
<img width="633" height="479" alt="image" src="https://github.com/user-attachments/assets/ba2c2c22-0ffb-469f-b19a-303d94d69d22" />

