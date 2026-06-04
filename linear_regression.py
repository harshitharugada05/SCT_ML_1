import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("Housing.csv")

# Features
X = data[["area", "bedrooms", "bathrooms"]]

# Target
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Example prediction
import pandas as pd

new_house = pd.DataFrame({
    "area": [2000],
    "bedrooms": [3],
    "bathrooms": [2]
})

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])