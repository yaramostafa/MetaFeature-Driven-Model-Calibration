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
## 📬 ** Testing using Postman**

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
        "Stddev Kurtosis of Numerical Features": 0.1952716167592422
    }
}
```

---

## 📊 **Send the Request**

1. Click on **Send** in Postman.
2. If successful, you'll receive a response similar to:

```json
{
    "0": {
        "ELASTICNETCV": 0.12,
        "HUBERREGRESSOR": 0.15,
        "LASSO": 0.18,
        "LinearSVR": 0.20,
        "QUANTILEREGRESSOR": 0.25,
        "XGBRegressor": 0.10
    }
}
```

- Each classifier will return a probability value.

---
