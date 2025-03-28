import torch
from torch.nn import functional as F
from model import GPTLanguageModel  # Must match your model file

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load vocabulary and helpers (must match training time)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Load model
model = GPTLanguageModel()
model.load_state_dict(torch.load('gpt_language_model.pt', map_location=device))
model.to(device)
model.eval()

# 🔁 Loop for user input
print("Type a prompt and hit Enter (type 'exit' to quit)\n")
while True:
    prompt = input("📝 Prompt: ")
    if prompt.lower() in {'exit', 'quit'}:
        print("Goodbye!")
        break

    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=300)

    print("\n🧠 Generated Text:\n")
    print(decode(output[0].tolist()))
    print("-" * 60)
