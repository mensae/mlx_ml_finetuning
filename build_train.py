import json
import os
import re
from sklearn.model_selection import train_test_split

VER = "v3"

RAW_DATA_PATH = f"data/raw/messages_{VER}.json"
TRAIN_JSONL_PATH = f'data/for_train/{VER}/train.jsonl'
VALID_JSONL_PATH = f'data/for_train/{VER}/valid.jsonl'

PROMPT_Q = """
Sei un utente nella chat di Twitch di nome Minnarone. Io ti fornirò i messaggi attualmente presenti in chat, e tu devi produrre il prossimo.
Produci un solo singolo messaggio, ponendo una domanda. Il tuo messaggio deve rispecchiare lo stile di quelli precedenti in chat. 
Rifletti attentamente sullo stile (evita emoji e prediligi le emote di Twitch).\n\n
"""
PROMPT_B = """
Sei un utente in una chat di Twitch di nome Minnarone. Io ti fornirò i messaggi attualmente presenti in chat, e tu devi produrre il prossimo.
Produci un solo singolo messaggio. Il tuo messaggio deve rispecchiare lo stile di quelli precedenti in chat.
Rifletti attentamente sullo stile (evita emoji e prediligi le emote di Twitch). \n\n
"""

def main():
	
	print(f"Chosen version {VER}.")
	
	"""Main function to orchestrate the data processing pipeline."""
	print("Loading and cleaning raw data...")
	raw_data = get_raw_data_clean()
	print(f"Loaded {len(raw_data)} entries")
	
	print("Filtering data...")
	filtered_data = filter_data(raw_data)
	print(f"Filtered down to {len(filtered_data)} entries")
	
	print("Converting to prompt-completion format...")
	converted_data, n_questions, n_base = convert_to_prompt_completion(filtered_data)
	print(f"Converted {len(converted_data)} entries")
	print(f"\t -{n_base} base prompts")
	print(f"\t -{n_questions} questions prompt")
	
	if converted_data:
		print("\nSample entry:")
		sample = converted_data[0]
		print(f"Prompt: {sample['prompt'][:100]}...")
		print(f"Completion: {sample['completion'][:50]}...")
	
	print("\nSplitting and saving data...")
	split_and_save_data(converted_data)

def clean_text(text):
	"""Clean text by removing non-ASCII characters."""
	return text.encode('ascii', 'ignore').decode()

def is_number(text):
	"""Check if a given text consists only of digits."""
	return re.fullmatch(r'\d+', text) is not None

def filter_data(data):
	"""Filter data to remove entries where at least 80% of previous messages are numbers."""
	filtered_data = []
	
	for entry in data:
		previous_messages = entry['previous_messages']
		
		if not previous_messages:
			continue
		
		num_count = sum(1 for msg in previous_messages if is_number(msg))
		num_ratio = num_count / len(previous_messages)
		
		if num_ratio < 0.8:
			filtered_data.append(entry)
	
	return filtered_data

def get_raw_data_clean(file_path=RAW_DATA_PATH):
	"""
	Retrieve data from raw json and clean it.
	
	Args:
		file_path: Path to the JSON file containing raw data
		
	Returns:
		List of cleaned data entries
	"""
	with open(file_path) as f:
		data = json.load(f)
		for entry in data:
			entry['previous_messages'] = [clean_text(msg) for msg in entry['previous_messages']]
			entry['message'] = clean_text(entry['message'])
	
	return data

def convert_to_prompt_completion(data):
	"""
	Convert the data to prompt-completion format required by mlx_lm.
	
	Args:
		data: List of data entries with 'previous_messages' and 'message' keys
		
	Returns:
		List of dictionaries with 'prompt' and 'completion' keys
	"""
	new_data = []
	n_questions = 0
	for d in data:
		
		is_question = d['message'].endswith("?")
		
		n_questions += 1 if is_question else 0
		prompt = PROMPT_Q if is_question else PROMPT_B # select which prompt
		prompt += '\n'.join(f"- {msg}" for msg in d['previous_messages'])
		
		completion = d['message']
		new_data.append({'prompt': prompt, 'completion': completion})
	
	return new_data, n_questions, len(data)-n_questions

def split_and_save_data(data, train_path=TRAIN_JSONL_PATH, valid_path=VALID_JSONL_PATH, test_size=0.1, random_state=42):
	"""
	Split the data into train and validation sets and save to JSONL files.
	
	Args:
		data: List of dictionaries with 'prompt' and 'completion' keys
		train_path: Path to save the training data
		valid_path: Path to save the validation data
		test_size: Fraction of data to use for validation
		random_state: Random seed for reproducibility
	"""
	os.makedirs(os.path.dirname(train_path), exist_ok=True)
	
	train, valid = train_test_split(data, test_size=test_size, random_state=random_state)
	
	with open(train_path, 'w') as train_file:
		for item in train:
			train_file.write(json.dumps(item) + '\n')
	
	with open(valid_path, 'w') as valid_file:
		for item in valid:
			valid_file.write(json.dumps(item) + '\n')
	
	print(f"Data split and saved: {len(train)} training samples, {len(valid)} validation samples")
	print(f"Files saved at {train_path} and {valid_path}")

if __name__ == "__main__":
	main()
