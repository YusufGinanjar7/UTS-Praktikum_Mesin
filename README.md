# UTS-Praktikum_Mesin

# UTS Praktikum Pembelajaran Mesin  
## Klasifikasi Kelayakan Kredit Komputer dengan Naive Bayes (NIM Genap)

## Tahapan Pembuatan Model

### 1. Import Library dan Load Data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
```

### 2. Pengecekan Data
- Cek nilai null: `df.isnull().sum()`
- Cek duplikasi: `df.duplicated().sum()`
- Cek outlier pada fitur numerik

### 3. Preprocessing
- Encode fitur kategorikal menggunakan One Hot Encoding
- Pisahkan fitur (`X`) dan target (`y`)
- Split data: `train_test_split(X, y, test_size=0.2)`

### 4. Penanganan Imbalanced Dataset
Distribusi target `Buys_Computer` tidak seimbang:
- 1 : 669 (layak)
- 0 : 331 (tidak layak)

Dilakukan **oversampling** terhadap kelas minoritas (0) pada data training menggunakan `resample()`.

### 5. Pembuatan dan Pelatihan Model
Gunakan algoritma Naive Bayes:
```python
model = GaussianNB()
model.fit(X_train, y_train)
```

### 6. Evaluasi Model
- Predict: `y_pred = model.predict(X_test)`
- Evaluasi: `accuracy_score`, `classification_report`
- Visualisasi Confusion Matrix

### 7. Cross Validation
Gunakan 5-fold cross-validation untuk mengevaluasi kestabilan model:
```python
scores = cross_val_score(model, X, y, cv=5)
```

---

## Hasil Evaluasi

- **Accuracy**: 78%
- **F1-Score (Layak)**: 0.83
- **F1-Score (Tidak Layak)**: 0.70
- **Cross Validation Accuracy**: 77.10% ± 3.56%

---


