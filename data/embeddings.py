from transformers import AutoTokenizer, AutoModel
import torch

# Load the MathBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT-custom")
model = AutoModel.from_pretrained("tbs17/MathBERT-custom")

def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # Compute the mean of the last hidden states
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()
