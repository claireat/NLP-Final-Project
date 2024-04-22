from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

dataset = load_dataset("csv", data_files="ExtractedTweets.csv", split="train")
dataset = dataset.train_test_split(test_size=0.3)
print(dataset)
#print(dataset["train"][100])

tokenizer = AutoTokenizer.from_pretrained("kornosk/polibertweet-mlm")


def tokenize_function(examples):
    return tokenizer(examples["Tweet"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

