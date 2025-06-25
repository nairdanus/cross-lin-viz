from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json
import os

import config

from config import MODEL_CONFIG, ModelConfig



class ModelTester:

    def __init__(
                self, 
                model_name: str, 
                dataset_name: str,
                dataset_dir: str = "",
                dataset_split: str = "test",
                cpu : bool = False,
                model_config: ModelConfig = ModelConfig()):
                
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = load_dataset(dataset_name, dataset_dir)[dataset_split]

        self.config = model_config
        
        self.few_shots, self.excluded_datapoints = self.create_fewshots(model_config.number_few_shots)

        self.answer_path = "_".join(["ANS", model_name.split("/")[1], dataset_name.split("_")[-1].upper(), dataset_dir, ".json"])

        if not os.path.exists(self.answer_path):
            with open(self.answer_path, "w") as f:
                json.dump({
                    "config": model_config.to_dict(),
                    "model": model_name,
                    "answers": dict(),
                }, f)

    def __str__(self):
        return f"""
--------------------------------        
Model Tester {self.answer_path}
--------------------------------
"""


    def generate_text(self, prompt: str):
        """
        Generate text based on the prompt using model's configuration.
        """
        # Tokenize the prompt text
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        inputs.to(self.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=self.config.max_len,
            num_return_sequences=1,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def create_fewshots(self, n):
        few_shots = ""
        excluded = []
        for _ in range(self.few_shots):
            shot = random.choice(self.dataset)
            few_shots += f"Q: {shot['Q']}\nA: {shot['A']}\n\n"
            excluded.append(shot["id"])
        return few_shots, excluded


    def format_prompt(self, question):
        prompt = SYSTEM_PROMPT + "\n"
        prompt += self.few_shots
        prompt += f"Q: {question}\nA: "
        return prompt


    def test(self):
            """
            This method will prompt the model on dataset examples (with a few-shot setup).
            """
            for data_point in self.dataset:

                with open(self.answer_path, "r") as f:
                    answers = json.load(f)
            
                ident = str((data_point["id"], data_point["Q"]))

                prompt = self.format_prompt(data_point["Q"])

                if ident in answers["answers"]:
                    generated_text = answers["answers"][ident]
                else:
                    generated_text = self.generate_text(prompt)
                    answers["answers"][ident] = generated_text

                with open(self.answer_path, "w") as f:
                    json.dump(answers, f)



if __name__ == "__main__":

    config = MODEL_CONFIG

    tester = ModelTester(
        model_name = "Qwen/Qwen2.5-3B-Instruct", 
        dataset_name = "nairdanus/VEC_prompt_en",
        dataset_dir = "color",
        model_config=config)
    
    print(tester)

    tester.test()
    