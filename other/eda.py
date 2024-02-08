import os
import matplotlib.pyplot as plt
import pandas as pd


# root_dir = '/lustre/ai/dataset/dip/CCSN_v2/data'  # Replace with the actual path to your root directory

# class_counts = {}  # Dictionary to store class names and their corresponding image counts

# for subdir in os.listdir(root_dir):
#     class_dir = os.path.join(root_dir, subdir)
#     if os.path.isdir(class_dir):
#         class_name = subdir
#         num_images = len(os.listdir(class_dir))  # Count the images in the class directory
#         class_counts[class_name] = num_images

# # Create a bar chart
# plt.figure(figsize=(10,6))
# bars = plt.bar(class_counts.keys(), class_counts.values(), color='orange')
# plt.xlabel('Classes')
# plt.ylabel('Number of Images')
# plt.title('Number of Images per Class')
# plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', color='black', size=10)



# # Save the figure
# plt.savefig('clean_class_distribution.png')  # You can change the file name and format as needed


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

import splitfolders
splitfolders.ratio('/home/dip_21/project/cloud/images', output="/home/dip_21/project/data", seed=786, ratio=(0.85,0.15))