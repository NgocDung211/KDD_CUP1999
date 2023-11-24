import numpy as np
import matplotlib.pyplot as plt

names = ["MLPClassifier", "RandomForestClassifier","LogisticRegression", "DecisionTreeClassifier",
           "SVC"]
kernal_evals = dict()

TPR = [98.74, 91.17, 99.46, 91.55, 85.18 ]
TNR = [91.27, 91.50, 46.34, 91.41, 87.76 ]
# TPR1 = [97.65, 97.12, 93.56, 97.14, 91.51]
# FNR1 = [33.56, 36.05, 50.08, 37.64, 33.73]
for i, name in enumerate(names):
  kernal_evals[str(name)] = [TPR[i], 100 - TNR[i]]

keys = [key for key in kernal_evals.keys()]
values = [value for value in kernal_evals.values()]
fig, ax = plt.subplots(figsize=(20, 6))
ax.bar(np.arange(len(keys)) - 0.2,
       [value[0] for value in values], color='darkred', width=0.25, align='center')
ax.bar(np.arange(len(keys)) + 0.2,
       [value[1] for value in values], color='y', width=0.25, align='center')
for i, value in enumerate(values):
    ax.text(i - 0.2, value[0] + 0.01, f"{value[0]:.2f}",
            ha='center', va='bottom', color='black')
    ax.text(i + 0.2, value[1] + 0.01, f"{value[1]:.2f}",
            ha='center', va='bottom', color='black')
ax.legend(["TNR", "FNR"])
ax.set_xticklabels(keys)
ax.set_xticks(np.arange(len(keys)))
plt.ylabel("Accuracy")
plt.show()
# 2
# Thời gian 481.65s
# this is the result of MLPClassifier
# TPR: 67.83
# TNR: 97.24
# FNR: 2.76