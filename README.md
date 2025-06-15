# Vehicle-Maintenance-Predictor
An AI-based web application that predicts whether a vehicle requires maintenance, using real-world usage data and health indicators. Built with XGBoost and deployed using Streamlit, the system analyzes critical parameters like mileage, engine temperature, oil level, tire thread depth, brake wear, and service history to make accurate predictions.

Key Features:
  Predicts maintenance needs using a machine learning classification model,
   Uses MinMaxScaler to normalize vehicle parameter inputs,
   Built and trained using XGBoost with over 200 labeled samples,
   Achieves up to 97% accuracy on validation data,
   Web interface built with Streamlit for easy user input and real-time predictions,
   Debug mode shows raw/scaled inputs and model confidence,
   Deployed on Streamlit Cloud for instant access and testing.

Tech Stack:
  Python, Pandas, NumPy,
   XGBoost, Scikit-learn,
   Streamlit,
   Joblib (for model & scaler persistence).

How It Works:
  User enters vehicle usage and health parameters into the app,
   Input data is scaled using the same scaler used during training,
   The trained model predicts whether maintenance is required,
   The app displays a clear result along with prediction confidence.

!!!Will upload code shortly!!!
