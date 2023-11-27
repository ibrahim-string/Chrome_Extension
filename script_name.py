from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the domain of your extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
print("hello243")
@app.post("http://127.0.0.1:8000/summarize")
def summarize_text(text: str):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    print("hello")
    # Generate summary
    summary_ids = model.generate(**inputs)

    # Decode the summary and return it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}
@app.post("http://127.0.0.1:8000/close_model")
def close_model():
    # Close the model explicitly
    model.close()
    return {"message": "Model closed successfully"}








