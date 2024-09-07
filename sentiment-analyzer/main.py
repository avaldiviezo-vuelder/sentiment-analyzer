from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

model_name = './models/nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

app = FastAPI()

class TextData(BaseModel):
    text: str

@app.post("/analyze")
async def analuze_sentiment(data: TextData):
    try:
        inputs = tokenizer(
            data.text, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment = model.config.id2label[scores.argmax().item()]
        return {"sentiment": sentiment}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TrainData(BaseModel):
    example: str
    response: str

@app.post("/train")
async def train_sentiment(data: TrainData):
    try:
        example = data.example
        response = data.response
        
        labels = [0 if response == "1 star" else 1 if response == "2 stars" else 2 if response == "3 stars" else 3 if response == "4 stars" else 4]

        encodings = tokenizer(example, truncation=True, padding=True, return_tensors="pt")

        train_dataset = Dataset.from_dict({"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels})

        training_args = TrainingArguments(
            output_dir = model_name,
            num_train_epochs = 3,
            per_device_train_batch_size = 4,
            warmup_steps=10,
            weight_decay = 0.01,
            logging_dir = "./logs",
            logging_steps = 10,
            save_steps=10,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)

        return {"status": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))