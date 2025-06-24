# 🚗 Car Price Detection Using Machine Learning

## 🧠 Project Overview

This project aims to **predict the price of used cars** using **Machine Learning (ML)** models. The system takes features like car brand, model year, mileage, fuel type, engine power, and other specifications as input and predicts a fair market price.

This can be used in:
- Online car marketplaces
- Used car valuation tools
- Automotive industry analytics
- Personal decision-making for buyers/sellers

The project leverages **regression algorithms**, **data preprocessing**, and **feature engineering** to build an accurate and interpretable price prediction system.

---

## 🎯 Objectives

1. Collect and preprocess real-world used car data.
2. Explore and visualize relationships between car features and price.
3. Train and evaluate multiple regression models (e.g., Linear Regression, Random Forest).
4. Predict car prices based on user inputs.
5. Deploy a simple web interface or API for practical use.

---

## 🧰 Technologies Used

- **Python 3.x**
- **Pandas / NumPy**: For data manipulation
- **Scikit-learn**: For ML modeling
- **Matplotlib / Seaborn**: For visualization
- **Streamlit / Flask / FastAPI**: For building a web app
- **Jupyter Notebook**: For development and analysis

---

## 📁 Dataset

### Sample Source:
[Kaggle - Used Car Price Prediction Dataset](https://www.kaggle.com/datasets)

### Example Features:

| Feature | Description |
|--------|-------------|
| `name` | Name and model of the car |
| `year` | Year of manufacture |
| `selling_price` | Target variable (price) |
| `km_driven` | Kilometers driven |
| `fuel` | Fuel type (Petrol, Diesel, etc.) |
| `seller_type` | Individual or Dealer |
| `transmission` | Manual or Automatic |
| `owner` | Number of previous owners |
| `engine` | Engine displacement (cc) |
| `max_power` | Power output (bhp) |
| `mileage` | Fuel efficiency (kmpl or mpg) |

---

## 🔬 Methodology

### Step 1: Data Loading & Preprocessing

```python
import pandas as pd

df = pd.read_csv('car_data.csv')
df.drop(['torque'], axis=1, inplace=True)
df['year'] = 2023 - df['year']  # Convert to age
```

### Step 2: Feature Encoding

```python
df = pd.get_dummies(df, drop_first=True)
```

### Step 3: Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop('selling_price', axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 4: Model Training

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
```

### Step 5: Evaluation

```python
from sklearn.metrics import mean_absolute_error, r2_score

preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
print("R² Score:", r2_score(y_test, preds))
```

---

## 🧪 Results

| Model | MAE | R² Score |
|-------|-----|----------|
| Linear Regression | ₹85,000 | 0.76 |
| Decision Tree | ₹50,000 | 0.89 |
| **Random Forest** | **₹35,000** | **0.94** |
| Gradient Boosting | ₹40,000 | 0.92 |

> *Note: Values are sample-based and may vary with dataset.*

---

## 📈 Sample Output

### Input:
```
Name: Maruti Swift VDI
Year: 2018
Kilometers Driven: 45000
Fuel: Diesel
Seller Type: Individual
Transmission: Manual
Owner: First Owner
Engine: 1248 cc
Max Power: 74 bhp
Mileage: 28.4 kmpl
```

### Predicted Selling Price:
```
₹495,000
```

---

## 🚀 Future Work

1. **Web App Interface**: Build a Streamlit/Flask app for users to enter car details and get predictions.
2. **Mobile App**: Integrate with Android/iOS apps for on-the-go pricing.
3. **Deep Learning Models**: Use neural networks for improved accuracy.
4. **Real-Time Scraping**: Scrape online car listing sites for up-to-date data.
5. **Explainable AI (XAI)**: Add SHAP values to explain feature importance.

---

## 📦 Folder Structure

```
car-price-detection/
│
├── data/
│   └── car_data.csv        # Raw dataset
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── models/
│   └── car_price_model.pkl # Saved trained model
│
├── app/
│   └── app.py              # Web application script
│
├── utils/
│   ├── preprocessing.py    # Data cleaning functions
│   └── evaluation.py       # Metric calculation
│
├── README.md
└── requirements.txt
```

---

## 📚 References

1. Scikit-learn Documentation – https://scikit-learn.org/stable/
2. Pandas Documentation – https://pandas.pydata.org/docs/
3. Kaggle Car Dataset – https://www.kaggle.com/datasets
4. Streamlit Docs – https://docs.streamlit.io/

---

## ✅ License

MIT License – see `LICENSE` for details.

---

Would you like me to:
- Generate the full Python script (`car_price_prediction.py`)?
- Include a Jupyter Notebook version?
- Provide instructions for deploying this as a web app?

Let me know how I can help further! 😊
