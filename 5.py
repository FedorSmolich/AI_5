import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Генерация тестовых данных
np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 12))
Y = np.random.randint(0, 2, size=(100, 1))
Y = np.hstack([Y, np.logical_not(Y).astype(int)])

# Сохранение данных в файлы
np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')

# Делим на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Создание модели
model = keras.Sequential([keras.layers.Dense(12, input_shape=(12,), activation='sigmoid'), keras.layers.Dense(2, activation='softmax')])

# Компиляция модели с categorical_crossentropy и метрикой accuracy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test))

# Оцениваем качество
y_pred = np.argmax(model.predict(X_test), axis=1)

y_true = np.argmax(Y_test, axis=1)
print("Точность:", accuracy_score(y_true, y_pred))

# Визуализация потерь обучения и проверки
plt.plot(history.history['loss'], label='Потеря обучения')
plt.plot(history.history['val_loss'], label='Потеря проверки')
plt.xlabel('Эпохи')
plt.ylabel('Потеря')
plt.legend()
plt.title('Потеря по эпохам')
plt.show()


Y_train_labels = np.argmax(Y_train, axis=1)
Y_test_labels = np.argmax(Y_test, axis=1)

# Логистическая регрессия
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train_labels)
y_pred_lr = lr.predict(X_test)
print("Точность логистической регрессии:", accuracy_score(Y_test_labels, y_pred_lr))

# Случайный лес
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train_labels)
y_pred_rf = rf.predict(X_test)
print("Точность случайного леса:", accuracy_score(Y_test_labels, y_pred_rf))