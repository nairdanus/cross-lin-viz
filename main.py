import ollama
from collections import defaultdict
import random
import os

SYSTEM_PROMPT = """
You are an expert in object recognition.
Using your expertise, always answer faithfully about the visual and embodied attributes of objects.
Always make sure to keep the answer the same way as in provided examples.
Always make sure to answer in the same language in which a question is formulated: If a question is formulated in language L, answer only in language L.
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
        for l in f.readlines():
            _id, entry = l.split(",")
            if _id in forbidden_idx:
                continue
            p, a = entry.replace("Q: ", "").strip().split(" A: ")
            yield _id, p, prompt('llama3:instruct', few_shots, p), a


if __name__ == "__main__":

    # llama3:instruct, llava-llama3:latest 
    for response in main("es", "mass", "llava-llama3:latest"):
        print(response)
