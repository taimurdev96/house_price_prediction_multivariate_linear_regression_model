import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_excel("housing_data.xlsx")
df.fillna({"Bedrooms": 6}, inplace = True)
df["Price"] = df["Price"].fillna(df["Price"].mean())
print(df)
linear_regression = linear_model.LinearRegression()
linear_regression.fit(df[["Area", "Bedrooms", "Distance"]].values, df.Price)
print("Model's accuracy is:", linear_regression.score(df[["Area", "Bedrooms", "Distance"]].values, df.Price))
print("Model's co-efficient is:", linear_regression.coef_)
print("Model's intercept is:", linear_regression.intercept_)
print("Model's predicted value based on a certain input:", linear_regression.predict([[927, 21, 8]]))
predicted = (4550.98213037 * 927) + (108975.19976861 * 21) + (21887.50130916 * 8) + 1680131.176307661
print("Predicted value as computed by the equation y = m * x + b:", predicted)
scatter = plt.scatter(
    x = df["Area"],
    y = df["Price"],
    c = df["Bedrooms"],
    s = df["Distance"],
    cmap = 'viridis',
    alpha = 0.6
)
plt.colorbar(scatter, label = "Bedrooms")
plt.xlabel("Area", color = 'blue')
plt.ylabel("Price", color = 'green')
plt.title("Area vs. Price (color = Bedrooms, size = Distance)")
plt.show()