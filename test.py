from transformers import AutoModel, AutoTokenizer, pipeline
import torch

# Choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select mode path here
pretrained_LM_path = "kornosk/polibertweet-mlm"

# Load model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModel.from_pretrained(pretrained_LM_path)

inputs = tokenizer('dog', return_tensors="pt")
outputs = model(**inputs)

inputs1 = tokenizer('cat', return_tensors="pt")
outputs1 = model(**inputs1)


inputs2 = tokenizer('oahdaohiaoshdasd', return_tensors="pt")
outputs2 = model(**inputs2)
#pprint.pp(outputs.pooler_output[0])

dog = outputs.pooler_output[0]
cat = outputs1.pooler_output[0]
random = outputs2.pooler_output[0]

def cosine_similarity(v1, v2):
    dotprod = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(a * a for a in v1) ** 0.5
    mag2 = sum(b * b for b in v2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0
    return dotprod / (mag1 * mag2)

print("dog + cat", cosine_similarity(dog, cat))
print("dog + random", cosine_similarity(dog, random))
print("random + cat", cosine_similarity(random, cat))

