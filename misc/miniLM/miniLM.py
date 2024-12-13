from transformers import AutoTokenizer, AutoModel

# Baixar o tokenizer e o modelo diretamente do Hugging Face
tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384", use_fast=True)
model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

# Teste simples
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

print("Hello, world! embeddings shape:")
print(outputs.last_hidden_state.shape)

