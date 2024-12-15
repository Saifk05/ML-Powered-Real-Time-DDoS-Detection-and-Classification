DDoS Detection and Traffic Monitoring System
This project implements a DDoS Detection System using Machine Learning models to classify network traffic and detect potential DDoS attacks. It features:

1. Data preprocessing and scaling using scikit-learn
2. Model training and evaluation for Random Forest, Logistic Regression, and Neural Network classifiers
3. Accuracy comparison across models and saving the best-performing model
4. A simple web-based traffic monitoring dashboard (HTML, CSS, JavaScript)
   
Features
1. Machine Learning Models:
2. Random Forest
3. Logistic Regression
4. Neural Network (MLP Classifier)
5. Real-time Monitoring Dashboard:
5. Web dashboard to visualize traffic data

Dataset:
1. Custom dataset for training and testing models
2. Accuracy Comparison:
3. Model accuracy comparison summary
4. Best Model Saving:
5. Automatically saves the best model as a .pkl file

Prerequisites
To run this project, ensure you have the following installed:
1. Python 3.8 or higher
2. Git
3. Pip (Python package manager)
4. Virtual Environment (recommended)


How to Clone the Repository
1. Open your terminal or command prompt.
2. Navigate to the directory where you want to clone the project.
3. Run the following command:
4. Copy code
5. git clone https://github.com/Saifk05/ML-Powered-Real-Time-DDoS-Detection-and-Classification.git
6. cd DDOS


Setting Up the Project
Create a Virtual Environment:
python -m venv .venv

Activate the Virtual Environment:

On Windows:
.\.venv\Scripts\activate

On macOS/Linux:
source .venv/bin/activate

Install Required Dependencies:

Install all dependencies from the requirements.txt file:
pip install -r requirements.txt
Dependencies

The following libraries are required and will be installed:

1. pandas - Data manipulation
2. numpy - Numerical operations
3. matplotlib - Data visualization
4. seaborn - Enhanced visualizations
5. scikit-learn - Machine learning library
6. flask - Web framework for the dashboard

Run the Project
Train the Models: Run the train_model.py script to train the models and save the best-performing model.
python Python\ book/train_model.py

Run the Web Dashboard: Navigate to the app directory and start the Flask server:
cd app
python app.py
Open the Dashboard: Open your browser and go to: http://127.0.0.1:5000

Testing and Outputs
The terminal will display model accuracy:

1. Random Forest Accuracy: 0.90
2. Logistic Regression Accuracy: 0.87
3. Neural Network Accuracy: 0.85

Best Model: Random Forest with Accuracy: 0.90

The Flask dashboard will show network traffic data and predictions.
