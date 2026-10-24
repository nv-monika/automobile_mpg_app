# Automobile Fuel Efficiency (MPG) Predictor

A Streamlit machine-learning app that predicts MPG from vehicle features. It includes the uploaded Auto MPG dataset, automatic preprocessing, three regression algorithms, cross-validation model selection, interactive Plotly charts, and a prediction form.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Use another dataset

Upload a CSV or Excel file from the sidebar. Choose the numeric target column (normally `mpg`). The remaining columns are treated as input features. Missing numeric and categorical feature values are imputed in the training pipeline.

## Models compared

- Linear Regression
- Random Forest Regressor
- Extra Trees Regressor

The app selects the model with the highest mean cross-validation R² score and reports holdout test metrics. The best model is used by the vehicle prediction form.
