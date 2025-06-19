from datasets import load_dataset, Dataset
import random
import os
from tqdm import tqdm
import json
import os

from time import sleep
from typing import Union, Dict, List, Tuple

import deepl

from huggingface_hub import login

from decouple import config as secret_config

hf_token = secret_config("HF")
deepl_token = secret_config("DEEPL")

login(hf_token)
deepl_client = deepl.DeepLClient(deepl_token)


if not os.path.exists("translations.json"):
    TRANSLATION_DICT = {}
else:
    with open("translations.json", "r") as f:
        TRANSLATION_DICT = json.load(f)

import config
TGT_LANGS = config.TGT_LANGS

def translate_func(
        text: str,
        source_language: str,
        target_language: str,
        backup_translator=None,
) -> str:
    while True:
        try:
            translation = deepl_client.translate_text(text,
                                               source_lang=source_language,
                                               target_lang=target_language)
            return translation
        except AttributeError as ae:
            print(f"Errors in translating {text}: {ae}")
            print(f"Retry...")


def translate_vec_dataset(
    dataset: List[Tuple[int, str]],
    target_languages: List[str],
    concept_category: str,
    translate_function=translate_func,
) -> Dict[str, List]:
    translated_ds = {}
    orig_ds = dataset

    for tl in target_languages:

        if not tl in TRANSLATION_DICT:
            TRANSLATION_DICT[tl] = dict()

        instances = []
        for i in tqdm(
                range(len(orig_ds[concept_category])),
                desc=f"Translating {concept_category} to {tl}"):

            small_dict = dict()
            for k ,v in orig_ds[concept_category][i].items():
                if not isinstance(v, str) or k == "relation":
                    translated_v = v
                elif v in TRANSLATION_DICT[tl]:
                    translated_v = TRANSLATION_DICT[tl][v]
                else:
                    translated_v = translate_function(
                        text=v, source_language="en", target_language=tl
                    ).text
                    TRANSLATION_DICT[tl][v] = translated_v
                    with open("translations.json", "w") as f:
                        json.dump(TRANSLATION_DICT, f)
                    sleep(1)

                small_dict[k] = translated_v


            instances.append(small_dict)

        with open("backup_" + tl + "_" + concept_category + ".json", "w") as f:
            json.dump(instances, f)


        hf_ds = Dataset.from_list(instances)
        hf_ds.push_to_hub("nairdanus/multilingual_vec_dataset_" + tl, concept_category, split="test")
        sleep(5)
        

def generate_multilingual_vec_dataset(vec_ds):
    for c in vec_ds:
        translated_ds = translate_vec_dataset(
            dataset=vec_ds,
            target_languages=TGT_LANGS,
            concept_category=c,
        )


def load_en():
    def load_en_concept(concept: str):    
        if concept not in ['color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness']:
            raise ValueError("Please provide one of the following concepts: 'color', 'size', 'shape', 'height', 'material', 'mass', 'temperature' or 'hardness'")
        ds = load_dataset("tobiaslee/VEC", concept)
        ds = ds['test']
        return ds
    data = dict()
    for c in tqdm(['color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness'], desc="Generating en dataset: "):
        ds = load_en_concept(c)
        data[c] = ds

        # ds.push_to_hub("nairdanus/multilingual_vec_dataset_en", c, split="test")
    return data

if __name__ == "__main__":
    en_data = load_en()
    generate_multilingual_vec_dataset(en_data)
