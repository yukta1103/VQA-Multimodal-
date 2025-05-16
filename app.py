import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import gradio as gr
from utils import load_model, answer_question

# Load model and processor
processor, model = load_model()

def vqa_pipeline(image: Image.Image, question: str) -> str:
    return answer_question(image, question, processor, model)

# Gradio UI
demo = gr.Interface(
    fn=vqa_pipeline,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Question")],
    outputs=gr.Textbox(label="Answer"),
    title="Visual Question Answering - BLIP-2",
    description="Upload an image and ask a question about it. Powered by BLIP-2 and HuggingFace."
)

if __name__ == "__main__":
    demo.launch()
