## **1. Prerequisites**

Make sure you have:

- **Python 3.x** installed
- Flask installed (`pip install flask`)
- Joblib installed (`pip install joblib`)
- The model files:
   - `final_model.pkl`
   - `temperature_scaler.pkl`
- **Postman** installed on your system
---
## **2. Start the Flask Server**

1. Save the provided Flask code in a file named `app.py`.
2. Run the server:
   ```bash
   python app.py
   ```
3. The server will start on:
   ```
   http://127.0.0.1:5000
   ```
---
## **3. Testing using Postman**

### **Endpoint URL**
```
POST http://127.0.0.1:5000/predict
```

### **Headers**
Add the following headers:
- **Content-Type**: `application/json`

### **Body**
Choose **raw** and set the type to **JSON**. Use the following sample JSON payload:

```json
{
    "0": {
        "num_clients": 10.0,
        "Sum of Instances in Clients": 13821.0,
        "Max. Of Instances in Clients": 1383.0,
        "Min. Of Instances in Clients": 1382.0,
        "Stddev of Instances in Clients": 0.3,
        "Average Dataset Missing Values %": 4.992465884583631,
        "Min Dataset Missing Values %": 4.121475054229935,
        "Max Dataset Missing Values %": 5.571635311143271,
        "Stddev Dataset Missing Values %": 0.4489697353421885,
        "Average Target Missing Values %": 4.992465884583631,
        "Min Target Missing Values %": 4.121475054229935,
        "Max Target Missing Values %": 5.571635311143271,
        "Stddev Target Missing Values %": 0.4489697353421885,
        "No. Of Features": 3.0,
        "No. Of Numerical Features": 3.0,
        "No. Of Categorical Features": 0.0,
        "Sampling Rate": 0.1666666666666666,
        "Average Skewness of Numerical Features": 0.072566663,
        "Minimum Skewness of Numerical Features": 1.1289151943589609e-05,
        "Maximum Skewness of Numerical Features": 1.305305017292974,
        "Stddev Skewness of Numerical Features": 0.2456136575521862,
        "Average Kurtosis of Numerical Features": 1.3473565423922125,
        "Minimum Kurtosis of Numerical Features": 0.3190575980238463,
        "Maximum Kurtosis of Numerical Features": 1.502076620762341,
        "Stddev Kurtosis of Numerical Features": 0.1952716167592422,
        "Avg No. of Symbols per Categorical Features": 0.0,
        "Min. No. Of Symbols per Categorical Features": 0.0,
        "Max. No. Of Symbols per Categorical Features": 0.0,
        "Stddev No. Of Symbols per Categorical Features": 0.0,
        "Avg No. Of Stationary Features": 1.0,
        "Min No. Of Stationary Features": 0.0,
        "Max No. Of Stationary Features": 2.0,
        "Stddev No. Of Stationary Features": 0.4472135954999579,
        "Avg No. Of Stationary Features after 1st order": 2.2,
        "Min No. Of Stationary Features after 1st order": 1.0,
        "Max No. Of Stationary Features after 1st order": 3.0,
        "Stddev No. Of Stationary Features after 1st order": 0.6,
        "Avg No. Of Stationary Features after 2nd order": 2.9,
        "Min No. Of Stationary Features after 2nd order": 2.0,
        "Max No. Of Stationary Features after 2nd order": 3.0,
        "Stddev No. Of Stationary Features after 2nd order": 0.3,
        "Avg No. Of Significant Lags in Target": 0.0,
        "Min No. Of Significant Lags in Target": 0.0,
        "Max No. Of Significant Lags in Target": 0.0,
        "Stddev No. Of Significant Lags in Target": 0.0,
        "Avg No. Of Insignificant Lags in Target": 0.0,
        "Max No. Of Insignificant Lags in Target": 0.0,
        "Min No. Of Insignificant Lags in Target": 0.0,
        "Stddev No. Of Insignificant Lags in Target": 0.0,
        "Avg. No. Of Seasonality Components in Target": 2.0,
        "Max No. Of Seasonality Components in Target": 2.0,
        "Min No. Of Seasonality Components in Target": 2.0,
        "Stddev No. Of Seasonality Components in Target": 0.0,
        "Average Fractal Dimensionality Across Clients of Target": 0.009828662,
        "Maximum Period of Seasonality Components in Target Across Clients": 13.0,
        "Minimum Period of Seasonality Components in Target Across Clients": 2.0,
        "Entropy of Target Stationarity": 0.3250829733914482
    }
}
```

---

## **4. Send the Request**

1. Click on **Send** in Postman.
2. If successful, you'll receive a response similar to:

```json
{
    "0": {
        "ELASTICNETCV": 7.944891119829134e-10,
        "HUBERREGRESSOR": 0.09174518476290221,
        "LASSO": 0.04749160848035781,
        "LinearSVR": 0.19160172766619232,
        "QUANTILEREGRESSOR": 0.015493822448004764,
        "XGBRegressor": 0.6536676558480538
    }
}
```

- Each classifier will return a probability value.

---
