from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./models/" + model_name)
tokenizer.save_pretrained("./models/" + model_name)

print("Modelo y tokenizador descargados y guardados localmente")