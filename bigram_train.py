import torch
import torch.nn as nn
from torch.nn import functional as F

# Read and process the input text file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create character mappings
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # Convert text to list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # Convert list of integers to text

# Encode the dataset into a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split into training and validation sets (90% train, 10% validation)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Parameters
block_size = 8  # Context length for predictions
batch_size = 32  # Training batch size

# Function to get random training batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Define the Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # Take last time step
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append sampled token
        return idx

# Execute only if the script is run directly
if __name__ == "__main__":
    torch.manual_seed(1337)  # For reproducibility

    # Initialize model
    vocab_size = len(chars)
    model = BigramLanguageModel(vocab_size)

    # Optimizer for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    for step in range(10000):  # Increase steps for better results
        xb, yb = get_batch('train')  # Get batch
        logits, loss = model(xb, yb)  # Compute loss
        optimizer.zero_grad(set_to_none=True)  # Reset gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    print(f"Final Training Loss: {loss.item()}")

    # Generate new text
    idx = torch.zeros((1, 1), dtype=torch.long)
    generated_text = decode(model.generate(idx, max_new_tokens=100)[0].tolist())

    print("\nGenerated Text:\n")
    print(generated_text)
