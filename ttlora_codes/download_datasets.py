from datasets import load_dataset

# Load the MNLI dataset from the GLUE benchmark
datasets = load_dataset('squad')

# Select a subset of the dataset (e.g., the first 1000 examples for training and validation)
train_dataset = datasets['train'].select(range(1000))
val_dataset = datasets['validation_matched'].select(range(1000))