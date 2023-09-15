from sklearn.dummy import DummyRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)
from dummy_regressor_with_fracsum import DummyRegressorWithFracSum

# Загрузим датасет по раку молочной железы
# Он представляет собой классический и очень простой набор данных для бинарной классификации.
data = load_breast_cancer()
X = data.data
y = data.target

# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создадим DummyRegressor с параметром strategy='mean'
dummy_mean = DummyRegressor(strategy="mean")
dummy_mean.fit(X_train, y_train)

# Создадим DummyRegressor с параметром strategy='median'
dummy_median = DummyRegressor(strategy="median")
dummy_median.fit(X_train, y_train)

# Создадим DummyRegressor с параметром strategy='median'
dummy_fracsum = DummyRegressorWithFracSum(strategy="fracsum")
dummy_fracsum.fit(X_train, y_train)

# Предскажем значения на тестовом наборе данных
y_pred_mean = dummy_mean.predict(X_test)
y_pred_median = dummy_median.predict(X_test)
y_pred_fracsum = dummy_fracsum.predict(X_test)

# Вычислим метрики для оценки качества модели
mse_mean = mean_squared_error(y_test, y_pred_mean)
mae_mean = mean_absolute_error(y_test, y_pred_mean)
medae_mean = median_absolute_error(y_test, y_pred_mean)

mse_median = mean_squared_error(y_test, y_pred_median)
mae_median = mean_absolute_error(y_test, y_pred_median)
medae_median = median_absolute_error(y_test, y_pred_median)

mse_fracsum = mean_squared_error(y_test, y_pred_fracsum)
mae_fracsum = mean_absolute_error(y_test, y_pred_fracsum)
medae_fracsum = median_absolute_error(y_test, y_pred_fracsum)

# Выведем результаты
print(
    f"Dummy Regressor (mean) - MSE: {mse_mean}, MAE: {mae_mean}, Median AE: {medae_mean}"
)
print(
    f"Dummy Regressor (median) - MSE: {mse_median}, MAE: {mae_median}, Median AE: {medae_median}"
)
print(
    f"Dummy Regressor (fracsum) - MSE: {mse_fracsum}, MAE: {mae_fracsum}, Median AE: {medae_fracsum}"
)
