from huggingface_hub import login
import torch
from PIL import Image
import os
from fastapi import FastAPI, UploadFile, File
from peft import PeftModel, PeftConfig
from transformers import PaliGemmaProcessor, AutoModelForPreTraining

hf_token = os.getenv('HF_KEY')

login(token= hf_token)

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = PeftConfig.from_pretrained("Mohit1Kulkarni/paligemma_Graph_Desc")
base_model = AutoModelForPreTraining.from_pretrained("google/paligemma-3b-pt-224").to(device)
model = PeftModel.from_pretrained(base_model, "Mohit1Kulkarni/paligemma_Graph_Desc").to(device)

processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")

@app.post("/analyze/")
async def analyze_image(input_text: str, image: UploadFile = File(...)):
    image = Image.open(image.file).convert("RGB")

    # Preprocessing Inputs
    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    inputs = processor(text=input_text, images=image, padding="longest",
                       do_convert_rgb=True, return_tensors="pt").to(device)
    inputs = inputs.to(dtype=model.dtype)

    # Generating and Decoding Output
    with torch.no_grad():
        output = model.generate(**inputs, max_length=496)

    response = processor.decode(output[0], skip_special_tokens=True)
    return {"result": response}