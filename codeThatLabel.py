import spacy
import jsonlines

# Load the trained model
model_path = "./my_trained_model"
nlp = spacy.load(model_path)

# Load the unlabeled data
unlabeled_data_file = "data/homework2_train.jsonl"

# Open the JSONL file and classify each record
classified_data = []
with jsonlines.open(unlabeled_data_file) as reader:
    for record in reader:
        text = record["text"]
        doc = nlp(text)
        predicted_labels = doc.cats
        classified_data.append({"text": text, "predicted_labels": predicted_labels})

# Optionally, you can save the classified data to a file or process it further
output_file = "classified_data.jsonl"
with jsonlines.open(output_file, mode="w") as writer:
    writer.write_all(classified_data)