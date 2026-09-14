import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Data: apartment size (m²) -> monthly rent (CHF) ---
#X = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60,
#              65, 70, 75, 80, 85, 90, 95, 100, 105]).reshape(-1, 1)
#y = np.array([620, 910, 780, 1080, 950, 1310, 1180, 1290, 1620, 1490, 1840, 1700, 2050, 1880, 2210, 1990, 2380, 2150])

# Use random numbers
rng = np.random.default_rng(42)          # fixed seed = reproducible
X = np.arange(20, 106, 5).reshape(-1, 1)
noise = rng.normal(0, 300, size=X.shape[0])   # 180 = spread in CHF
y = 20 * X.ravel() + 300 + noise

# --- 2. Split into training and test data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. Fit the model ---
model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_
print(f"rent = {slope:.2f} * size + {intercept:.2f}")

# --- 4. Evaluate on unseen data ---
y_pred = model.predict(X_test)
print(f"R²   = {r2_score(y_test, y_pred):.3f}")
print(f"RMSE = {mean_squared_error(y_test, y_pred) ** 0.5:.2f} CHF")

# --- 5. Predict something new ---
print(f"48 m² -> {model.predict([[48]])[0]:.0f} CHF")

# --- 6. Plot ---
line_x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
line_y = model.predict(line_x)

plt.scatter(X_train, y_train, color="steelblue", label="training data")
plt.scatter(X_test, y_test, color="darkorange", marker="s", label="test data")
plt.plot(line_x, line_y, color="crimson", linewidth=2, label="regression line")

# show residuals (the errors the model minimizes)
for xi, yi in zip(X_train.ravel(), y_train):
    plt.plot([xi, xi], [yi, model.predict([[xi]])[0]],
             color="gray", linestyle=":", linewidth=1)

plt.xlabel("size (m²)")
plt.ylabel("rent (CHF)")
plt.legend()
plt.title("Linear Regression Demo")
plt.tight_layout()
plt.show()


