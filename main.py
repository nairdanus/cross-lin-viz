import ollama
from collections import defaultdict
import random
import os
from tqdm import tqdm

SYSTEM_PROMPT = """
You are an expert in object recognition.
Using your expertise, always answer faithfully about the visual and embodied attributes of objects.
Always make sure to keep the answer the same way as in provided examples.
Answer in the same language in which the question was asked, unless explicitly instructed otherwise. If the input is in Spanish, respond in Spanish; if the input is in Korean, respond in Korean, etc.
Do not default to English unless the query is presented in English.
"""

def format_few_shots(path: str, amount: int):
    messages = []
    with open(path, "r") as f:
        lines = f.readlines()

    idx = [random.randint(0, len(lines)-1) for _ in range(amount)]

    for l in [lines[i] for i in idx]:
        idx, prompt = l.split(',')
        q, a = prompt.strip().replace('Q: ', '').lstrip().split('A: ')
        messages.extend([
            {
                'role': 'user',
                'content': q
            },
            {
                'role': 'assistant',
                'content': a
            }
        ])
    return messages, idx

def prompt(model, few_shots, question):
    response = ollama.chat(
            model = model,
            messages= [{'role': 'system', 'content': SYSTEM_PROMPT}] + few_shots + [{'role': 'user', 'content': question}]
        )
    return response['message']['content']


def main(tgt_lan: str, concept: str, model: str):

    few_shots, forbidden_idx = format_few_shots(os.path.join("dataset", "en", f"{concept}.csv"), 30)

    with open(os.path.join("dataset", tgt_lan, f"{concept}.csv")) as f:
        for l in tqdm(f.readlines(), desc=tgt_lan+" - "+concept):
            _id, entry = l.split(",")
            if _id in forbidden_idx:
                continue
            p, a = entry.replace("Q: ", "").strip().split(" A: ")
            yield _id, p, prompt('llama3:instruct', few_shots, p), a

def all_concepts(tgt_lan: str, model: str):
    if not os.path.exists('results'):
        os.mkdir('results')
    dir_name = os.path.join('results', model)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    dir_name = os.path.join(dir_name, tgt_lan)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for c in ['color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness']:
        file_name = os.path.join(dir_name,c + ".txt")
        if os.path.exists(file_name):
            print("Hallo, das gibt es schon!")
            continue
        
        for response in main(tgt_lan, c, model):
            with open(file_name, 'a') as f:
                f.write(str(response)+"\n")


if __name__ == "__main__":

    # llama3:instruct, llava-llama3:latest 
    for l in ['ko', 'de', 'zh-CN', 'es', 'ja']:
        all_concepts(tgt_lan=l, model="llama3:instruct")

    for l in ['ko', 'de', 'zh-CN', 'es', 'ja']:
        all_concepts(tgt_lan=l, model="llava-llama3")
