setup project structure

python setup_project.py

activate environment

source .venv/bin/activate

tensorboard --logdir=modellogs

mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1  --port 5000 