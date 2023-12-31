Реализована новая стратегия “fracsum” для DummyRegressor, которая возвращает сумму дробных частей в обучающем наборе аналогично тому, как реализованы стратегии “mean” и “median”.

а) Такой регрессор способен делать множественную регрессию, так как он просто возвращает сумму дробных частей целевых переменных, независимо от числа объясняющих переменных.

б) Он способен делать множественную регрессию, так как не зависит от числа объясняющих переменных.

в) Не нужно оборачивать его в sklearn.multioutput.MultiOutputRegressor, так как он уже способен обрабатывать многомерные (многозначные) целевые переменные без дополнительных оберток.


# DummyRegressorWithFracSum

`DummyRegressorWithFracSum` - это расширение класса `DummyRegressor` из библиотеки scikit-learn, которое добавляет новую стратегию "fracsum". Эта стратегия предсказывает сумму дробных частей значений обучающей выборки, аналогично стратегиям "mean" и "median". Этот репозиторий содержит код для `DummyRegressorWithFracSum`, а также примеры его использования.



```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Использование

``` 
python3 main.py
```

Пример использования `DummyRegressorWithFracSum`:


```
python
import numpy as np
from sklearn.dummy import DummyRegressorWithFracSum

# Создаем регрессор с стратегией "fracsum"
regressor = DummyRegressorWithFracSum(strategy='fracsum')

# Обучаем регрессор на данных
X = np.array([[1, 2], [2, 3], [3, 4]])  # Объясняющие переменные
y = np.array([0.5, 1.3, -0.8])  # Целевые переменные
regressor.fit(X, y)

# Получаем результат
print("Коэффициент:", regressor.constant_)
```