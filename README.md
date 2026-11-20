# \# Credit Scoring MLOps — PD Prediction

# 

# End-to-end educational MLOps project for predicting the probability of default (PD) using the \*\*UCI Default of Credit Card Clients\*\* dataset (30,000 observations).

# 

# \*\*Target:\*\* `default.payment.next.month`  

# \*\*Target interpretation:\*\* `1` = default in the following month, `0` = no default.

# 

# This project demonstrates reproducible ML training, experiment tracking, data validation, automated testing, model serving, and basic drift monitoring.

# 

# \*\*Disclaimer:\*\* This is an educational project, not a production-ready credit decision system. Real-world deployment would require independent validation, fairness assessments, regulatory compliance, and ongoing monitoring.

# 

# \## Architecture

# 

# ```text

# UCI Credit Card Dataset

# &#x20;        |

# &#x20;        v

# Pandera Data Validation

# &#x20;        |

# &#x20;        v

# Stratified Train/Test Split (80/20)

# &#x20;        |

# &#x20;        v

# Feature Engineering and Preprocessing

# &#x20;        |

# &#x20;        v

# Five MLflow Cross-Validation Experiments

# &#x20;        |

# &#x20;        v

# GridSearchCV Hyperparameter Tuning

# &#x20;        |

# &#x20;        v

# Holdout Evaluation and ROC Curve

# &#x20;        |

# &#x20;        v

# Trained Model (joblib) + MLflow

# &#x20;        |

# &#x20;        v

# FastAPI + Docker

# &#x20;        |

# &#x20;        v

# PSI Drift Monitoring

# ```

# 

# \*\*Technologies:\*\* Python 3.11, pandas, scikit-learn, Pandera, MLflow, DVC, DagsHub, FastAPI, Docker, pytest, Black, Flake8, and GitHub Actions.

# 

# \## Project Structure

# 

# | Directory / File | Description |

# |---|---|

# | `data/raw/` | Original UCI dataset, tracked with DVC |

# | `data/processed/` | Reproducible training and test datasets |

# | `src/data/` | Data preparation and Pandera validation |

# | `src/features/` | Feature engineering |

# | `src/models/` | Training, evaluation, tuning, and inference |

# | `src/api/` | FastAPI prediction service |

# | `src/monitoring/` | PSI drift monitoring |

# | `tests/` | Unit tests and API tests |

# | `notebooks/` | Exploratory data analysis |

# | `scripts/` | Utility and validation scripts |

# | `models/` | Trained model artifacts |

# | `reports/` | Evaluation figures and monitoring results |

# | `.github/workflows/` | GitHub Actions CI |

# | `dvc.yaml` | Reproducible pipeline stages |

# | `dvc.lock` | Locked pipeline dependencies and outputs |

# | `metrics.json` | Holdout evaluation metrics |

# 

# \## Installation

# 

# Requires \*\*Python 3.11\*\*, Git, and access to the project repository.

# 

# Clone the repository:

# 

# ```bash

# git clone https://github.com/tamararozhnova88-cyber/credit-scoring-model.git

# cd credit-scoring-model

# ```

# 

# Create a virtual environment.

# 

# \*\*Windows PowerShell:\*\*

# 

# ```powershell

# python -m venv .venv

# .\\.venv\\Scripts\\Activate.ps1

# ```

# 

# \*\*Linux/macOS:\*\*

# 

# ```bash

# python3 -m venv .venv

# source .venv/bin/activate

# ```

# 

# Install dependencies:

# 

# ```bash

# python -m pip install -r requirements.txt

# ```

# 

# \## Data Versioning with DVC and DagsHub

# 

# The project uses DVC to track datasets and generated model artifacts.

# 

# Remote storage is hosted on DagsHub:

# 

# https://dagshub.com/tamararozhnova88-cyber/credit-scoring-model

# 

# The DVC remote is already configured in `.dvc/config`. \*\*Do not run `dvc init` or `dvc add` after cloning.\*\*

# 

# Download tracked data and model artifacts:

# 

# ```bash

# dvc pull

# ```

# 

# The repository is configured to use the `dagshub` remote by default.

# 

# Check the remote configuration:

# 

# ```bash

# dvc remote list

# ```

# 

# Check pipeline status:

# 

# ```bash

# dvc status

# ```

# 

# Check remote synchronization:

# 

# ```bash

# dvc status -c

# ```

# 

# Downloading files from a public repository should not require write credentials. Uploading new DVC artifacts requires appropriate DagsHub permissions.

# 

# \*\*Security:\*\* Access tokens and passwords must not be committed to Git. Local credentials belong in `.dvc/config.local`, which must remain excluded from version control.

# 

# \## Data Validation

# 

# Pandera is used to validate the input dataset before training.

# 

# Run validation:

# 

# ```bash

# python -m scripts.validate\_data

# ```

# 

# The validation checks the dataset against the expected schema and rejects invalid data.

# 

# \## Reproducible ML Pipeline

# 

# The pipeline is defined in `dvc.yaml` and contains two stages:

# 

# 1\. \*\*prepare\*\* — validate and prepare the dataset, then create a deterministic stratified train/test split.

# 2\. \*\*train\*\* — run ML experiments, tune the model, evaluate the selected model, and generate artifacts.

# 

# Run the complete pipeline:

# 

# ```bash

# dvc repro

# ```

# 

# DVC uses dependency and output hashes to determine which stages need to be rerun.

# 

# Training can take several minutes because the pipeline includes five cross-validation experiments and a final hyperparameter search.

# 

# Generated artifacts include:

# 

# \- `models/credit\_default\_model.pkl`

# \- `metrics.json`

# \- `reports/figures/roc\_curve.png`

# \- Processed training and test datasets

# 

# The holdout test set is reserved for final evaluation after model selection.

# 

# \## Experiment Tracking with MLflow

# 

# MLflow tracks model experiments and evaluation metrics.

# 

# The project uses a local SQLite tracking database:

# 

# ```text

# sqlite:///mlflow.db

# ```

# 

# Start the MLflow UI:

# 

# ```bash

# mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ```

# 

# Open:

# 

# http://127.0.0.1:5000

# 

# The training workflow includes five initial cross-validation experiments and a tuned final model.

# 

# MLflow tracking data is stored locally unless an external tracking server is configured.

# 

# \## Model Evaluation

# 

# The best selected model is a \*\*Gradient Boosting classifier\*\*.

# 

# \### Holdout Test Results

# 

# | Metric | Value |

# |---|---:|

# | ROC-AUC | 0.7736 |

# | Precision | 0.6714 |

# | Recall | 0.3560 |

# | F1-score | 0.4653 |

# 

# \*\*Cross-validation ROC-AUC:\*\* approximately \*\*0.7846\*\*.

# 

# ROC-AUC is used as the primary model selection metric. Precision, recall, and F1-score are calculated using a classification threshold of 0.5.

# 

# The ROC curve is saved to:

# 

# ```text

# reports/figures/roc\_curve.png

# ```

# 

# These results describe performance on the project's held-out test set and should not be interpreted as evidence of production readiness.

# 

# \## FastAPI Model Serving

# 

# The trained model is exposed through a REST API.

# 

# Start the API:

# 

# ```bash

# uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# ```

# 

# Open the interactive API documentation:

# 

# http://127.0.0.1:8000/docs

# 

# \### Endpoints

# 

# | Endpoint | Method | Description |

# |---|---|---|

# | `/health` | GET | Health check |

# | `/predict` | POST | Predict probability of default |

# 

# The `/predict` endpoint accepts a JSON object containing the 23 raw model input features, excluding `ID` and the target column.

# 

# The response includes:

# 

# \- `default\_prediction`: predicted class (`0` or `1`)

# \- `default\_probability`: estimated default probability between 0 and 1

# 

# Generate a sample request from the processed test data:

# 

# ```bash

# python -c "import pandas as pd,json; d=pd.read\_csv('data/processed/test.csv'); print(json.dumps(d.drop(columns='default.payment.next.month').iloc\[0].to\_dict()))"

# ```

# 

# \## Docker

# 

# The project includes a Dockerfile for containerized inference.

# 

# Ensure that the trained model is available locally before building the image. You can download it using `dvc pull` or generate it using `dvc repro`.

# 

# \*\*Windows PowerShell:\*\*

# 

# ```powershell

# .\\scripts\\docker\_build\_run.ps1

# ```

# 

# \*\*Linux/macOS:\*\*

# 

# ```bash

# bash scripts/docker\_build\_run.sh

# ```

# 

# Once the container is running, open:

# 

# http://localhost:8000/docs

# 

# \## Data Drift Monitoring

# 

# The project implements basic prediction-score drift monitoring using the \*\*Population Stability Index (PSI)\*\*.

# 

# With the API running:

# 

# ```bash

# python -m src.monitoring.drift --api-url http://localhost:8000 --sample-size 200

# ```

# 

# The monitoring script:

# 

# 1\. Sends sample test records to the prediction API.

# 2\. Collects predicted default probabilities.

# 3\. Compares them with reference training-score distributions.

# 4\. Calculates PSI using training-quantile bins.

# 5\. Saves results to `reports/drift.json`.

# 

# A PSI value of \*\*0.2 or greater\*\* triggers a warning.

# 

# This threshold is a heuristic rather than a regulatory standard.

# 

# The test dataset simulates incoming observations; the monitoring demonstration does not use a live production data stream.

# 

# \## Testing and Code Quality

# 

# Run unit and API tests:

# 

# ```bash

# pytest -q

# ```

# 

# Check formatting with Black:

# 

# ```bash

# black --check src tests scripts

# ```

# 

# Run Flake8:

# 

# ```bash

# flake8 src tests scripts --max-line-length=100 --extend-ignore=E203,W503

# ```

# 

# The project includes tests for data handling, validation, model-related functionality, and API behavior.

# 

# \## Continuous Integration

# 

# GitHub Actions automatically runs checks on pushes and pull requests.

# 

# The CI workflow includes:

# 

# \- Dependency installation

# \- Black formatting checks

# \- Flake8 linting

# \- pytest tests

# 

# The CI workflow does not require downloading the full DVC dataset or retraining the model.

# 

# Full data validation and model training are performed separately through the local DVC pipeline.

# 

# \## Reproducibility

# 

# To reproduce the project from a fresh clone:

# 

# ```bash

# git clone https://github.com/tamararozhnova88-cyber/credit-scoring-model.git

# cd credit-scoring-model

# python -m venv .venv

# ```

# 

# Activate the environment, then run:

# 

# ```bash

# python -m pip install -r requirements.txt

# dvc pull

# python -m scripts.validate\_data

# dvc repro

# pytest -q

# ```

# 

# Start MLflow or the FastAPI service as needed.

# 

# The DVC lockfile records pipeline dependencies and outputs, while the DagsHub remote stores tracked artifacts.

# 

# \## Limitations and Future Improvements

# 

# The project is intended as an educational demonstration.

# 

# Potential improvements include:

# 

# \- Probability calibration and decision-threshold optimization

# \- Fairness and bias assessment

# \- Data quality monitoring on incoming requests

# \- Automated drift alerts

# \- Centralized MLflow tracking

# \- Model registry and approval workflows

# \- Production secrets management

# \- Automated deployment

# \- Monitoring on live production data

# 

# \## Repository Links

# 

# \*\*GitHub:\*\* https://github.com/tamararozhnova88-cyber/credit-scoring-model

# 

# \*\*DagsHub:\*\* https://dagshub.com/tamararozhnova88-cyber/credit-scoring-model

# 

# \*\*Dataset:\*\* UCI Default of Credit Card Clients

# 

# \---

# 

# \*\*Project type:\*\* Educational end-to-end MLOps pipeline for credit default prediction.

