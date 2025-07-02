from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import load_dataset
import torch
import json
import os
import sys
import argparse
from tqdm import tqdm
import re
import guidance

from config import MODEL_CONFIG, ModelConfig


if not os.path.exists("RESULTS"):
    os.mkdir("RESULTS")

if not os.path.exists(os.path.join("RESULTS", "ANS")):
    os.mkdir(os.path.join("RESULTS", "ANS"))

if not os.path.exists(os.path.join("RESULTS", "METRICS")):
    os.mkdir(os.path.join("RESULTS", "METRICS"))


class ModelTester:

    def __init__(
                self, 
                model_name: str, 
                dataset_name: str,
                dataset_dir: str = "",
                dataset_split: str = "test",
                cpu : bool = False,
                system_prompt_path: str = "system_prompt.txt",
                few_shot_path: str = "few_shots.txt",
                model_config: ModelConfig = ModelConfig()):
                
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        self.dataset = load_dataset(dataset_name, dataset_dir)[dataset_split]
        self.all_gold = {d["A"] for d in self.dataset}

        self.config = model_config

        with open(system_prompt_path, "r") as f:
            self.system_prompt = f.read()
        if self.config.use_few_shots:
            with open (few_shot_path, "r") as f:
                self.system_prompt += "\n" + f.read()
        self.system_prompt += "\n### YOUR TASK\n"

        self.answer_file_name = "_".join(["ANS", model_name.split("/")[1], dataset_name.split("_")[-1].upper(), dataset_dir, "fewshot" if self.config.use_few_shots else "zeroshot"]) + ".json"
        self.answer_path = os.path.join("RESULTS", "ANS",  self.answer_file_name)

        if not os.path.exists(self.answer_path):
            with open(self.answer_path, "w") as f:
                json.dump({
                    "config": model_config.to_dict(),
                    "model": model_name,
                    "answers": dict(),
                }, f)

    def __str__(self):
        return f"""Model Tester {self.answer_path}"""

    def load_model(self):
        self.model = guidance.models.Transformers(self.model_name)


    def generate_text(self, prompt: str, choices: list):
        """
        Generate text based on the prompt using model's configuration.
        """

        if self.model is None:
            self.load_model()

        lm = self.model + prompt + guidance.select(choices)
        return str(lm)


    def format_prompt(self, question):
        prompt = self.system_prompt
        prompt += f"Q: {question}\nA: "
        return prompt


    def test(self):
        """
        This method will prompt the model with the entire dataset and return the metrics string.
        """
        gold_labels = []
        pred_labels = []

        with open(self.answer_path, "r") as f:
            answers = json.load(f)

        for i, data_point in tqdm(enumerate(self.dataset), desc="Iterating dataset", ncols=100, total=len(self.dataset)):

            ident = str((data_point["id"], data_point["Q"]))

            prompt = self.format_prompt(data_point["Q"])

            if ident in answers["answers"]:
                generated_text = answers["answers"][ident]
            else:
                choice1, choice2 = re.findall(r"('\w+')", prompt)[-2:]
                generated_text = self.generate_text(prompt, choices=[choice1, choice2])
                answers["answers"][ident] = generated_text

            if (i != 0 and i % 10 == 0) or i == len(self.dataset)-1:
                with open(self.answer_path, "w") as f:
                    json.dump(answers, f)

            try:
                pass
                # answer = generated_text[len(prompt):]
            except Exception as e:
                print("INDEX ERROR:", generated_text, len(prompt), len(generated_text))
                raise e

            pred = self.extract_answer(generated_text, data_point["A"])
            gold = data_point["A"].lower()

            if pred == "UNK":
                print("\n__________________________________")
                print("NO LABEL EXTRACTION:")
                print(generated_text)
                
            gold_labels.append(gold)
            pred_labels.append(pred)

            if (i != 0 and i % 50 == 0) or i == len(self.dataset)-1:
                print(f"\n\n___________________")
                print(f"Evaluation at step {i}")
                print(self.get_metrics(gold_labels, pred_labels))
                print(f"-----------------------")

        return self.get_metrics(gold_labels, pred_labels)


    def extract_answer(self, pred, gold):
        return pred.split()[-1].strip("'")
    

    def get_metrics(self, gold_labels, pred_labels):

        labels = list(self.all_gold)
        
        accuracy = accuracy_score(gold_labels, pred_labels)

        precision = precision_score(gold_labels, pred_labels, average="macro", zero_division=0)
        recall = recall_score(gold_labels, pred_labels, average="macro", zero_division=0)
        f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)

        precision_pc = precision_score(gold_labels, pred_labels, average=None, labels=labels, zero_division=0)
        recall_pc = recall_score(gold_labels, pred_labels, average=None, labels=labels, zero_division=0)
        f1_pc = f1_score(gold_labels, pred_labels, average=None, labels=labels, zero_division=0)

        precision_text = ", ".join([str(labels[i]) + ": " + str(precision_pc[i]) for i in range(len(labels))])
        precision_text = f"{precision}\n\t{precision_text}"
        recall_text = ", ".join([str(labels[i]) + ": " + str(recall_pc[i]) for i in range(len(labels))])
        recall_text = f"{recall}\n\t{recall_text}"
        f1_text = ", ".join([str(labels[i]) + ": " + str(f1_pc[i]) for i in range(len(labels))])
        f1_text = f"{f1}\n\t{f1_text}"

        conf_matrix = confusion_matrix(gold_labels, pred_labels)

        return f"""

----------------------------------
{self}
Metrics:
----
Accuracy: {accuracy}
Precision: {precision_text}
Recall: {recall_text}
F1: {f1_text}
----------------------------------

"""


def main(model_name, dataset_name, category):

    config = MODEL_CONFIG

    tester = ModelTester(
        model_name = model_name, 
        dataset_name = dataset_name,
        dataset_dir = category,
        model_config=config)
    
    print(tester)

    metrics = tester.test()

    with open(os.path.join("RESULTS", "METRICS", tester.answer_file_name.replace(".json", ".txt")), "w") as f:
        f.write(metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test a model on a dataset with a given category."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name or path of the model (e.g., Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset (e.g., nairdanus/VEC_prompt_en)"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="The category to test (e.g., color)"
    )
    args = parser.parse_args()

    try:
        main(
            model_name=args.model,
            dataset_name=args.dataset,
            category=args.category
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    