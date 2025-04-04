from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import numpy as np

def train_model(minnie_array, miyeon_array, shuhua_array, yuqi_array, soyeon_array):
    x = np.concatenate([minnie_array, miyeon_array, shuhua_array, yuqi_array, soyeon_array], axis=0)
    y = np.array([0] * len(minnie_array) + [1] * len(miyeon_array) + [2] * len(shuhua_array) + [3] * len(yuqi_array) + [4] * len(soyeon_array))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=16)

    model = SVC(C=10, kernel='rbf', gamma="scale", probability=True)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    return model, y_test, y_pred

def classification(model, y_test, y_pred):
    print(f'ความแม่นยำ : {accuracy_score(y_test, y_pred) * 100:.2f}%')
    print(classification_report(y_test, y_pred, target_names=["Minnie", "Miyeon", "Shuhua", "Yuqi", "Soyeon"]))

