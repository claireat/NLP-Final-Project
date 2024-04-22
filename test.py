from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, pipeline
import evaluate 
import numpy as np

dataset = load_dataset("csv", data_files="ExtractedTweets.csv", split="train")
dataset = dataset.train_test_split(test_size=0.3)
print(dataset)
#print(dataset["train"][100])

tokenizer = AutoTokenizer.from_pretrained("kornosk/polibertweet-mlm")


def tokenize_function(examples):
    return tokenizer(examples["Tweet"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
print(classifier(text))
[{'label': 'POSITIVE', 'score': 0.9994940757751465}]