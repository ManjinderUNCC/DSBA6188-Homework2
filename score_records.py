import spacy
from spacy.training import Example
import jsonlines
import random

# Load a blank English model
nlp = spacy.blank("en")

# Add text classification pipeline to the model
textcat = nlp.add_pipe('textcat_multilabel', last=True)
textcat.add_label("RELEVANT")
textcat.add_label("NOT_RELEVANT")

# Path to the processed data file
processed_data_file = "processed_data.jsonl"

# Open the JSONL file and extract text and labels
with jsonlines.open(processed_data_file) as reader:
    processed_data = list(reader)

# Convert processed data to spaCy format
spacy_train_data = []
for obj in processed_data:
    text = obj["text"]
    label = {"RELEVANT": obj["label"] == "RELEVANT", "NOT_RELEVANT": obj["label"] == "NOT_RELEVANT"}
    spacy_train_data.append(Example.from_dict(nlp.make_doc(text), {"cats": label}))

# Initialize the model and get the optimizer
optimizer = nlp.initialize()

# Train the text classification model
n_iter = 10
for i in range(n_iter):
    spacy.util.fix_random_seed(1)
    random.shuffle(spacy_train_data)
    losses = {}
    for batch in spacy.util.minibatch(spacy_train_data, size=8):
        nlp.update(batch, losses=losses, sgd=optimizer)
    print("Iteration:", i, "Losses:", losses)

# Save the trained model
output_dir = "./my_trained_model"
nlp.to_disk(output_dir)