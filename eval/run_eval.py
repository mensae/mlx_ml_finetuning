from mlx_lm import load, generate
import json
from tqdm import tqdm
import csv

VER = "v2"
OUT_CSV = f"{VER}_eval_out.csv"

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


def run_prompt(prompt, model, tokenizer):
	messages = [{"role": "user", "content": prompt}]
	prompt = tokenizer.apply_chat_template(
		messages, add_generation_prompt=True
	)
	
	text = generate(model, tokenizer, prompt=prompt, verbose=False)
	return text

with open("eval.json") as f:
	data = json.load(f)
	
print(f"Chosen version {VER}.")

print("Loading base model..")
model_base, tokenizer_base = load("../models/base/gemma-3-text-27b-it-4bit")
print("Loading finetuned model..")
model_ft, tokenizer_ft = load(f"../models/fused/fused-{VER}_gemma-3-text-27b-it-4bit")


# prepare OUT CSV
with open(OUT_CSV, 'w', newline='') as csvfile:
	csv_writer = csv.writer(csvfile)
	# Write the header
	csv_writer.writerow(['Streamer', '#', 'Base - Standard Prompt', 'Finetuned - Standard Prompt', 'Base - Question Prompt', 'Finetuned - Question Prompt'])

for streamer_key, messages in tqdm(data.items()):
	streamer = streamer_key.split("#")[0]
	entry_n = streamer_key.split("#")[1]
	messages = "\n".join([f"- {m}" for m in messages])
	
	filled_prompt_b = PROMPT_B + messages
	filled_prompt_q = PROMPT_Q + messages
	
	#answer_base_b = run_prompt(filled_prompt_b, model_base, tokenizer_base)
	#answer_base_q = run_prompt(filled_prompt_q, model_base, tokenizer_base)
	answer_base_q=""
	answer_base_b=""
	answer_ft_b = run_prompt(filled_prompt_b, model_ft, tokenizer_ft)
	answer_ft_q = run_prompt(filled_prompt_q, model_ft, tokenizer_ft)
	
	# Export in csv
	with open(OUT_CSV, 'a', newline='') as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerow([streamer, entry_n, answer_base_b, answer_ft_b, answer_base_q, answer_ft_q])
	
