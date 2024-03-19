import jsonlines

# Path to your dataset file
dataset_file = "exported_data.jsonl"

# Path to the output file
output_file = "processed_data.jsonl"

# Open the JSONL file and extract text and labels
try:
    with jsonlines.open(dataset_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in reader:
            text = obj.get("text")
            label = obj.get("accept", [])[0]  # Get the first accepted label if available
            if text and label:
                writer.write({"text": text, "label": label})
            else:
                print("Warning: Text or label missing in the JSON object.")
    print("Processing completed. Output written to:", output_file)
except Exception as e:
    print("Error:", e)
