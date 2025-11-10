# Diamond Price Prediction

A machine learning project for predicting diamond prices using various regression models. This project implements a complete ML pipeline with data ingestion, transformation, model training, and evaluation, following MLOps best practices.

## 🎯 Project Overview

This project predicts diamond prices based on their characteristics including carat, cut, color, clarity, and physical dimensions. The pipeline evaluates multiple regression models and automatically selects the best performer based on R² score.

## 🛠️ Tools & Technologies

### Core ML Stack
- **Python 3.9, 3.10, 3.11** - Multi-version support
- **scikit-learn** - Machine learning algorithms and preprocessing
- **pandas & numpy** - Data manipulation and numerical operations
- **XGBoost** - Gradient boosting implementation

### MLOps & Testing
- **Tox** - Automated testing across multiple Python versions
- **pytest** - Unit and integration testing framework
- **flake8** - Code style and quality checks
- **mypy** - Static type checking
- **GitHub Actions** - CI/CD pipeline automation

### Version Control & Tracking
- **DVC (Data Version Control)** - Data and model versioning
- **Git** - Source code version control

### Deployment
- **Docker** - Containerization for consistent environments
- **Docker Compose** - Multi-container orchestration

## 📁 Project Structure

```
DiamondPricePred/
│
├── .github/
│   └── workflows/
│       └── testing.yaml          # CI/CD pipeline configuration
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # Data loading and splitting
│   │   ├── data_transformation.py # Feature engineering
│   │   ├── model_trainer.py      # Model training and selection
│   │   └── model_evaluation.py   # Model evaluation metrics
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py  # End-to-end training workflow
│   │   └── prediction_pipeline.py # Inference pipeline
│   │
│   └── utils/
│       └── utils.py              # Helper functions
│
├── tests/
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── artifacts/                    # Generated artifacts (models, data)
├── pyproject.toml               # Project configuration
├── tox.ini                      # Tox configuration
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Git
- Docker (optional, for containerized deployment)
- DVC (optional, for data versioning)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AmrMohamed17/gemflow-diamond-price-mlops.git
cd gemflow-diamond-price-mlops
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e ".[dev]"
```

4. **Install DVC** (if using data versioning)
```bash
pip install dvc dvc-s3  # or dvc-gdrive, dvc-azure depending on your storage
```

## 📊 Using DVC for Data Version Control

DVC helps track and version your datasets and models alongside your code.

### Initial DVC Setup

1. **Initialize DVC**
```bash
dvc init
```

2. **Add remote storage** (example with S3)
```bash
# For AWS S3
dvc remote add -d myremote s3://mybucket/dvcstore

# For Google Drive
dvc remote add -d myremote gdrive://folder_id

# For local storage
dvc remote add -d myremote /path/to/local/storage
```

3. **Configure credentials**
```bash
# For S3
dvc remote modify myremote access_key_id 'your-access-key'
dvc remote modify myremote secret_access_key 'your-secret-key'
```

### Tracking Data with DVC

1. **Track your dataset**
```bash
dvc add artifacts/data.csv
git add artifacts/data.csv.dvc artifacts/.gitignore
git commit -m "Track raw data with DVC"
```

2. **Track trained models**
```bash
dvc add artifacts/model.pkl
dvc add artifacts/preprocessor.pkl
git add artifacts/model.pkl.dvc artifacts/preprocessor.pkl.dvc
git commit -m "Track trained models with DVC"
```

3. **Push data to remote storage**
```bash
dvc push
```

4. **Pull data from remote storage**
```bash
dvc pull
```

### DVC Pipeline Automation

Create a `dvc.yaml` file to define your ML pipeline:

```yaml
stages:
  data_ingestion:
    cmd: python src/pipeline/training_pipeline.py
    deps:
      - artifacts/data.csv
    outs:
      - artifacts/preprocessor.pkl
      - artifacts/model.pkl
      - artifacts/test.csv
```

Run the pipeline:
```bash
dvc repro
```

## 🐳 Docker Setup

### Building the Docker Image

1. **Create Dockerfile**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Expose port for API (if applicable)
EXPOSE 8000

# Run training pipeline
CMD ["python", "src/pipeline/training_pipeline.py"]
```

2. **Build the image**
```bash
docker build -t diamond-price-pred:latest .
```

3. **Run the container**
```bash
docker run -v $(pwd)/artifacts:/app/artifacts diamond-price-pred:latest
```

### Docker Compose for Complete Setup

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  training:
    build: .
    volumes:
      - ./artifacts:/app/artifacts
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    command: python src/pipeline/training_pipeline.py

  testing:
    build: .
    volumes:
      - .:/app
    command: tox
```

Run with Docker Compose:
```bash
docker-compose up training
docker-compose up testing
```

## 🧪 Testing

### Run all tests with Tox
```bash
tox
```

### Run specific test environments
```bash
# Test with Python 3.9
tox -e py39

# Test with Python 3.10
tox -e py310

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

### Code Quality Checks
```bash
# Linting
flake8 src --max-line-length=127

# Type checking
mypy src/ --ignore-missing-imports
```

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

- **Triggers**: Push to main, Pull Requests
- **Matrix Testing**: Tests across Ubuntu & Windows with Python 3.9, 3.10, 3.11
- **Checks**: Linting, type checking, unit tests, integration tests

## 📈 Model Training

Run the complete training pipeline:

```bash
python src/pipeline/training_pipeline.py
```

This will:
1. Load and split the data (80/20 train/test)
2. Apply transformations (encoding, scaling, imputation)
3. Train multiple models (Linear Regression, Decision Tree, XGBoost)
4. Select the best model based on R² score
5. Save the best model and preprocessor to `artifacts/`

## 📝 Models Evaluated

- **Linear Regression** - Baseline linear model
- **Decision Tree Regressor** - Non-linear tree-based model
- **XGBoost Regressor** - Gradient boosting ensemble

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🔗 Resources

- [DVC Documentation](https://dvc.org/doc)
- [Docker Documentation](https://docs.docker.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Tox Documentation](https://tox.wiki/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Predicting! 💎**