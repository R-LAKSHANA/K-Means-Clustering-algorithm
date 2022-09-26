# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:

### Step1
Import the necessary packages.

### Step2
Read the csv file using read_csv() and print the number of contents to be displayed using head().

### Step3
Scatter plot the ApplicantIncome and Loan Amount.

### Step4
Obtain the K-Mean clustering and print cluster center and labels using .cluster_centers_ and .labels_ respectively.

### Step5
Predict the cluster group for the ApplicantIncome and Loan Amount when it is 9200 and 110.

## Program:
```python
'''
Program to implement K-Means Clustering Algorithm.
Developed by: R LAKSHANA
RegisterNumber: 22004909
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('clustering.csv')

print(data.head(2))

x1 = data.loc[:,['ApplicantIncome','LoanAmount']]
print(x1.head(2))

X=x1.values
sns.scatterplot(X[:,0],X[:,1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()

kmean = KMeans(n_clusters = 4)
kmean.fit(X)

print('Cluster Centers: ',kmean.cluster_centers_)
print('Labels: ',kmean.labels_)

predicted_cluster = kmean.predict([[9200,110]])
print('The cluster group for the ApplicantIncome 9200 and Loan Amount 110 is ',predicted_cluster)
```
## Output:

![output](/OUTPUT.png)

## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program.