import os

import pandas as pd

clin = pd.read_excel("datasets/bcnb/patient-clinical-data.xlsx")
clin["ALN status"].value_counts()
# Imbalanced: 655, 210, 193

clin["Histological grading"].value_counts()
# Imbalanced: 1: 38, 2: 518, 3: 370
# 55% in the 2 category, only 4% in 1 category

