# Customer Churn Prediction MLOps

A complete MLOps pipeline for predicting customer churn using machine learning, built with FastAPI, MLflow, DVC, and Docker.

## 🚀 Features

- **Machine Learning Pipeline**: End-to-end ML workflow with data preprocessing, model training, and evaluation
- **Model Serving**: REST API for real-time churn predictions
- **Experiment Tracking**: MLflow integration for model versioning and metrics
- **Data Versioning**: DVC for dataset and artifact management
- **Containerization**: Docker support for easy deployment
- **Scalable Architecture**: Production-ready FastAPI application

## 🛠 Tech Stack

- **Python 3.10**
- **FastAPI** - Web framework for API
- **Scikit-learn** - Machine learning algorithms
- **MLflow** - Experiment tracking and model registry
- **DVC** - Data version control
- **Docker** - Containerization
- **Pandas & NumPy** - Data processing
- **XGBoost** - Gradient boosting algorithm
- **Imbalanced-learn** - Handling imbalanced datasets

## 📁 Project Structure

```
customer-churn-mlops/
├── app/
│   ├── main.py              # FastAPI application
│   └── __init__.py
├── src/
│   └── train.py             # Model training script
├── data/
│   ├── customer_churn.csv   # Dataset
│   └── customer_churn.csv.dvc
├── models/                  # Model artifacts (generated)
├── dvc-storage/            # DVC cache
├── mlruns/                 # MLflow experiments
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── dvc.yaml               # DVC pipeline
└── README.md
```

## 🏃‍♂️ Quick Start

### Prerequisites

- Docker and Docker Compose
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/AK-Jeevan/customer-churn-mlops.git
   cd customer-churn-mlops
   ```

2. **Set up environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python src/train.py
   ```

4. **Run the API locally**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t churn-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 churn-api
   ```

The API will be available at `http://localhost:8000`

## 📡 API Usage

### Health Check
```bash
GET /
```

### Churn Prediction
```bash
POST /predict
Content-Type: application/json

{
  "gender": "Male",
  "age": 35,
  "salary": 75000.0,
  "tenure": 5
}
```

**Response:**
```json
{
  "prediction": 0,
  "churn_probability": 0.23
}
```

- `prediction`: 0 (no churn) or 1 (churn)
- `churn_probability`: Probability of customer churning (0.0 to 1.0)

## 🔧 Development

### Training Pipeline

The training pipeline includes:
- Data preprocessing with scaling and encoding
- Handling imbalanced classes with SMOTE
- Hyperparameter tuning with GridSearchCV
- Model evaluation with cross-validation
- Artifact saving for deployment

### MLOps Workflow

1. **Data Management**: DVC tracks dataset versions
2. **Experiment Tracking**: MLflow logs parameters, metrics, and models
3. **Model Registry**: Best models are versioned and stored
4. **CI/CD**: Automated testing and deployment with Docker

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

AK Jeevan - [GitHub](https://github.com/AK-Jeevan)

Project Link: [https://github.com/AK-Jeevan/customer-churn-mlops](https://github.com/AK-Jeevan/customer-churn-mlops)