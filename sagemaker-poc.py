import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role
import pandas as pd
from datasets import load_dataset

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Load and preprocess the IMDb dataset
dataset = load_dataset("imdb")

# Function to preprocess the dataset
def preprocess_function(examples):
    return examples["text"], examples["label"]

# Preprocess the dataset
train_dataset = dataset["train"].map(preprocess_function, batched=True)
test_dataset = dataset["test"].map(preprocess_function, batched=True)

# Save datasets to CSV files
train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_dataset)

train_path = "train.csv"
test_path = "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# Upload datasets to S3
bucket = sagemaker_session.default_bucket()
prefix = "imdb-sentiment-analysis"

train_s3 = sagemaker_session.upload_data(train_path, bucket=bucket, key_prefix=f"{prefix}/train")
test_s3 = sagemaker_session.upload_data(test_path, bucket=bucket, key_prefix=f"{prefix}/test")

# Set up the Hugging Face estimator
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./scripts",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.17.0",
    pytorch_version="1.10.2",
    py_version="py38",
    hyperparameters={
        "epochs": 3,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "model_name": "distilbert-base-uncased",
        "output_dir": "/opt/ml/model"
    }
)

# Train the model
huggingface_estimator.fit({"train": train_s3, "test": test_s3})

# Deploy the model
huggingface_model = HuggingFaceModel(
    model_data=huggingface_estimator.model_data,
    role=role,
    transformers_version="4.17.0",
    pytorch_version="1.10.2",
    py_version="py38",
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)

# Test the deployed model
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def predict(text):
    encoded_input = tokenizer(text, return_tensors="pt")
    output = predictor.predict(encoded_input)
    return output

# Example prediction
review = "This movie was fantastic! I really enjoyed every moment of it."
result = predict(review)
print(f"Sentiment: {'Positive' if result[0][1] > result[0][0] else 'Negative'}")

# Clean up
predictor.delete_endpoint()
