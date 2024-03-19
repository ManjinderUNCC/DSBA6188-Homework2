import jsonlines

# Input file containing classified data
input_file = "classified_data.jsonl"

# Output file to store transformed data
output_file = "homework2_trainComplete.jsonl"

# Threshold for considering a label
threshold = 0.5

# Options for different categories
options = [
    {"id": "NOT_RELEVANT", "text": "NOT_RELEVANT", "meta": "0.00"},
    {"id": "RELEVANT", "text": "RELEVANT", "meta": "1.00"}
]

# Function to process each record
def process_record(record):
    # Extract text and predicted labels
    text = record["text"]
    predicted_labels = record["predicted_labels"]
    
    # Determine accepted categories based on threshold
    accepted_categories = [label for label, score in predicted_labels.items() if score > threshold]
    
    # Determine answer based on accepted categories
    answer = "accept" if accepted_categories else "reject"
    
    # Prepare options with meta
    options_with_meta = [
        {"id": option["id"], "text": option["text"], "meta": option["meta"]} for option in options
    ]
    
    # Construct the output record
    output_record = {
        "text": text,
        "cats": predicted_labels,
        "accept": accepted_categories,
        "answer": answer,
        "options": options_with_meta
    }
    
    return output_record

# Process input file and write transformed data to output file
with jsonlines.open(input_file, "r") as infile, jsonlines.open(output_file, "w") as outfile:
    for record in infile:
        output_record = process_record(record)
        outfile.write(output_record)
