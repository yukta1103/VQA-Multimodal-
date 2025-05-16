import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    return processor, model

def answer_question(image, question, processor, model):
    device = model.device
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)