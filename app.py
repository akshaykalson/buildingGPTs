from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from model import GPTLanguageModel

# === Load vocab and helpers ===
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# === Load model ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPTLanguageModel()
model.load_state_dict(torch.load("gpt_language_model.pt", map_location=device))
model.to(device)
model.eval()

# === FastAPI Setup ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# In-memory chat history
chat_history = []

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chat": chat_history})

@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, prompt: str = Form(...)):
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=300)

    full_text = decode(output[0].tolist())
    generated = full_text[len(prompt):]

    chat_history.append((prompt, generated))
    return templates.TemplateResponse("index.html", {"request": request, "chat": chat_history})
