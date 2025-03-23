from mlx_lm import load, generate
import json

VER = "v3"

print(f"Chosen version {VER}.")

def get_valid():
	with open(f"data/for_train/{version}/valid.jsonl") as f:
		data = [json.loads(line) for line in f]
	return data

valid_examples = get_valid()

#model, tokenizer = load("models/base/gemma-3-text-27b-it-4bit")
model, tokenizer = load(f"models/fused/fused-{VER}_gemma-3-text-27b-it-4bit")

for i in range(30,40):
	ds_prompt = valid_examples[i]['prompt']
	ds_answer = valid_examples[i]['completion']
	messages = [{"role": "user", "content": ds_prompt}]
	prompt = tokenizer.apply_chat_template(
		messages, add_generation_prompt=True
	)

	text = generate(model, tokenizer, prompt=prompt, verbose=False)
	print(f"Prompt: {ds_prompt}  \n")
	print("------------------------------")
	print("Correct Answer from dataset:")
	print(ds_answer)
	print("Generated text:")
	print(text)
	print("***************************************************\n\n")