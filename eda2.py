import os
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('/home/dip_21/project/cloud/train.csv')

# Group by 'label' and count occurrences
class_distribution = df['label'].value_counts()

# Plot the class distribution
plt.figure(figsize=(10, 6))
bars = plt.bar(class_distribution.index, class_distribution.values)

plt.xlabel('Class Label')
plt.ylabel('Frequency')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', color='black', size=10)
plt.title('Class Distribution')
plt.savefig('/home/dip_21/project2/plot/classdist.jpg')
