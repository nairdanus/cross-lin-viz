from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json


class ModelConfig:
    def __init__(
        self, 
        max_len: int = 10, 
        number_few_shots: int = 5, 
        no_repeat_ngram_size: int = 2, 
        top_k: int = 50, 
        top_p: float = 0.95, 
        temperature: float = 0.7,
        do_sample: bool = True
    ):

        self.max_len = max_len
        self.number_few_shots = number_few_shots
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.do_sample = do_sample


class ModelTester:

    def __init__(
                self, 
                model_name: str, 
                dataset_name: str,
                dataset_dir: str = "",
                dataset_split: str = "test",
                model_config: ModelConfig = ModelConfig()):
                
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = load_dataset(dataset_name, dataset_dir)[dataset_split]

        self.config = model_config

        self.answer_path = "_".join(model_name.split("/")[1], dataset_name.split("_")[-1].upper(), dataset_dir)

        if not os.path.exists(self.answer_path):
            with open(self.answer_path, "w") as f:
                json.dump(f, {
                    "model_config": model_config,
                    "model": model_name,
                    "answers": dict(),
                })


    def generate_text(self, prompt: str):
        """
        Generate text based on the prompt using model's configuration.
        """
        # Tokenize the prompt text
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        inputs.to(self.device)

        outputs = self.model.generate(
            inputs,
            max_length=self.config.max_len,
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


    def test(self):
            """
            This method will prompt the model on dataset examples (with a few-shot setup).
            """
            for data_point in self.dataset:

                with open(self.answer_path, "r") as f:
                    answers = json.read(f)
            
                ident = (data_point["id"], data_point["Q"])

                if ident in answers["answers"]:
                    generated_text = answers["answers"][ident]
                else:
                    generated_text = self.generate_text(prompt)
                    answers["answers"][ident] = generated_text

                with open(self.answer_path, "w") as f:
                    json.dump(f, answers)



if __name__ == "__main__":

    config = ModelConfig({
        "max_len": 10, 
        "number_few_shots": 5, 
        "no_repeat_ngram_size": 2, 
        "top_k": 50, 
        "top_p": 0.95, 
        "temperature": 0.7,
        "do_sample": True
    })

    tester = ModelTester(
        model_name = "Qwen/Qwen2.5-3B-Instruct", 
        dataset_name = "nairdanus/VEC_prompt_en",
        dataset_dir = "color",
        model_config=config)

    tester.test()
    