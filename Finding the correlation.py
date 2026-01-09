import numpy as np
import matplotlib.pyplot as plt

x = np.array([-9, -7, -5, -3.5, -1.0, 1, 3, 5, 7.9, 9.9])  
y = np.array([4, 4.5, 3, 4, 1.0, 1.3, -2, -3.6, -4.9, -5.8])  

r = np.corrcoef(x, y)[0, 1]
print("Pearson's r:", r)

# Calculate the slope and intercept for the line of best fit
slope, intercept = np.polyfit(x, y, 1)

# Scatter plot
plt.scatter(x, y, color='blue', label='Data Points')

# Add the regression line
plt.plot(x, slope * x + intercept, color='red', label='Line of Best Fit')

plt.title('Scatter Plot of Data Points with Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()  
plt.savefig('correlation_plot.png')  # Save for report
plt.show()