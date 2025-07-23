
# 🎓 Student Performance Indicator

A machine learning web application that predicts student performance based on various features like demographics, academic history, and lifestyle factors. Built using Python, Scikit-learn, Flask, and deployed as a modular project structure.

---

## 📌 Project Overview

This project uses a machine learning pipeline to analyze and predict student performance. It includes data ingestion, preprocessing, model training, evaluation, and deployment through a Flask web interface.

---

## 🚀 Tech Stack

| Tool / Library      | Usage                           |
|---------------------|----------------------------------|
| Python              | Core programming language        |
| Pandas, NumPy       | Data manipulation                |
| Scikit-learn        | Machine Learning pipeline        |
| Flask               | Web app deployment               |
| Matplotlib, Seaborn | Visualization                    |
| HTML/CSS            | Frontend (basic UI)              |
| Git & GitHub        | Version control & hosting        |

---

## 📁 Project Structure<br>
```text
Student-performance-indicator/
│
├── artifacts/                  # Stores intermediate artifacts like data files, models, plots
│
├── src/                        # Source code for the ML pipeline
│   ├── components/             # Data ingestion, transformation, model trainer modules
│   ├── pipelines/              # Training and prediction pipeline logic
│   ├── utils/                  # Utility functions
│   ├── exception.py            # Custom exception class
│   ├── logger.py               # Logging module
│   └── main.py                 # Entry point for training/testing pipeline
│
├── templates/                  # HTML templates for Flask app
│   └── index.html              # Home page for the web UI
    └── Home.html
│
├── app.py                      # Flask app script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Files and folders to ignore in Git
```

---

## ⚙️ Features

- ✅ Data Ingestion & Cleaning
- ✅ Outlier Detection & Transformation
- ✅ Feature Scaling using PowerTransformer
- ✅ Model Training (Multiple Algorithms)
- ✅ Model Evaluation & Selection
- ✅ Flask Web App for Prediction
- ✅ Deployment Ready Code Structure

---

## 🧠 How to Use

### 1. Clone the Repo
```
git clone https://github.com/seelamdivya23/Student-performance-indicator.git
cd Student-performance-indicator
```
### 2. Create & Activate Virtual Environment
```
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac
```
### 3. Install Requirements
```
pip install -r requirements.txt
```
### 4. Run the Application
```
python app.py
```
# 📊 Results
 ## Best Accuracy: ~88% 

 # 🙋‍♀️ Author
 Divya Seelam <br>
📧 Email: divyaseelam83@gmail.com
# 🌟 Acknowledgements
## Dataset
The dataset used in this project is [Student Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) from Kaggle.


Scikit-learn & Flask Documentation
## ⭐️ Don't forget to star the repo if you find it useful!



