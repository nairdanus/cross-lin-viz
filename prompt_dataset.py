from datasets import load_dataset, Dataset
import random
import os
import sys
from tqdm import tqdm

from time import sleep

from huggingface_hub import login

from decouple import config as secret_config
hf_token = secret_config("HF")
login(hf_token)

import config
TGT_LANGS = config.TGT_LANGS
SEED = config.SEED


def translate_relation(rel, lan):
    match rel:
        case 'color':
            match lan:
                case 'ko':
                    return '색'
                case 'de':
                    return 'Farbe'
                case 'es':
                    return 'el color'
                case 'zh':
                    return '颜色'
                case 'ja':
                    return '色'
        case 'material': 
            match lan:
                case 'ko':
                    return '재질'
                case 'es':
                    return 'el material'
                case 'zh':
                    return '材料'
                case 'ja':
                    return '材質'
        case 'shape': 
            match lan:
                case 'ko':
                    return '모양'
                case 'de':
                    return 'Form'
                case 'es':
                    return 'la forma'
                case 'zh':
                    return '形状'
                case 'ja':
                    return '形'

    raise ValueError(f"No translation found for lan {lan} and rel {rel}!")


def prompt_format(e, lan):

    def make_question_and_answer(e: dict, question_format: str):

        swap = random.random() < 0.5

        if not swap:
            if "positive" in e:
                first, second = e["positive"], e["negative"]
            else:
                first, second = e["obj1"], e["obj2"]
        else:
            if "positive" in e:
                first, second = e["negative"], e["positive"]
            else:
                first, second = e["obj2"], e["obj1"]
        
        first, second = first.lower(), second.lower()

        q = question_format.format(first, second)

        if "label" in e:
            if e["label"]:
                correct = first if not swap else second
            else:
                correct = second if not swap else first
        
        else:
            correct = first if not swap else second

        return q, correct



    def write_en(e):
        match e['relation']:
            case 'temperature':
                question_format = "What is hotter: '{}' or '{}'?"
            case 'size':
                question_format = "What is bigger: '{}' or '{}'?"
            case 'mass':
                question_format = "What is heavier: '{}' or '{}'?"
            case 'height':
                question_format = "What is taller: '{}' or '{}'?"
            case 'hardness':
                question_format = "What is harder: '{}' or '{}'?" 
            case 'color':
                question_format = "What is the color of '" + e['obj'] + "': '{}' or '{}'?"
            case 'material':
                question_format = "What is the material of '" + e['obj'] + "': '{}' or '{}'?"
            case 'shape':
                question_format = "What is the shape of '" + e['obj'] + "': '{}' or '{}'?"
            case _:
                raise ValueError("Wrong relation!")

        return make_question_and_answer(e, question_format)
    
    def write_ko(e): # TODO
        if "obj1" in e: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"'{e['obj1']}'가 '{e['obj2']}'보다 더 뜨겁나요?", f"{'예' if e['label'] else '아니요'}"
                case 'size':
                    return f"'{e['obj1']}'가 '{e['obj2']}'보다 큰가요?", f"{'예' if e['label'] else '아니요'}"
                case 'mass':
                    return f"'{e['obj1']}'가 '{e['obj2']}'보다 무겁나요?", f"{'예' if e['label'] else '아니요'}"
                case 'height':
                    return f"'{e['obj1']}'가 '{e['obj2']}'보다 키가 큰가요?", f"{'예' if e['label'] else '아니요'}"
                case 'hardness':
                    return f"'{e['obj1']}'가 '{e['obj2']}'보다 더 무겁나요?", f"{'예' if e['label'] else '아니요'}"
                case _:
                    raise ValueError("Wrong relation!")
                
        else: # color, material, shape
            if random.random() > 0.5:
                return f"{e['obj']}의 {translate_relation(e['relation'], lan)}은 무엇인가요: '{e['positive']}' 또는 '{e['negative']}'?", f"{e['positive']}" 
            else:
                return f"{e['obj']}의 {translate_relation(e['relation'], 'ko')}은 무엇인가요: '{e['negative']}' 또는 '{e['positive']}'?", f"{e['positive']}" 
    
    def write_de(e):
        match e['relation']:
            case 'temperature':
                question_format = "Was ist heißer: '{}' oder '{}'?"
            case 'size':
                question_format = "Was ist größer: '{}' oder '{}'?"
            case 'mass':
                question_format = "Was ist schwerer: '{}' oder '{}'?"
            case 'height':
                question_format = "Was ist höher: '{}' oder '{}'?"
            case 'hardness':
                question_format = "Was ist härter: '{}' oder '{}'?"
            case 'color':
                question_format = "Welche Farbe hat '" + e['obj'] + "': '{}' oder '{}'?"
            case 'material':
                question_format = "Aus welchem Material besteht '" + e['obj'] + "': '{}' oder '{}'?"
            case 'shape':
                question_format = "Welche Form hat '" + e['obj'] + "': '{}' oder '{}'?"
            case _:
                raise ValueError("Wrong relation!")

        return make_question_and_answer(e, question_format)


    def write_zh(e):
        match e['relation']:
            case 'temperature':
                question_format = "哪个比较热，'{}'还是'{}'?"
            case 'size':
                question_format = "哪个比较大，'{}'还是'{}'?"
            case 'mass':
                question_format = "哪个比较重，'{}'还是'{}'?"
            case 'height':
                question_format = "哪个比较高，'{}'还是'{}'?"
            case 'hardness':
                question_format = "哪个比较硬，'{}'还是'{}'?"
            case 'material':
                question_format = "'" + e['obj'] + "'是什么材料: '{}'还是'{}'?"
            case 'color':
                question_format = "'" + e['obj'] + "'是什么颜色: '{}'还是'{}'?"
            case 'shape':
                question_format = "'" + e['obj'] + "'是什么形状: '{}'还是'{}'?"
            case _:
                raise ValueError("Wrong relation!")

        return make_question_and_answer(e, question_format)
        
    
    def write_es(e):
        match e['relation']:
            case 'temperature':
                question_format = "¿Cuál es más caliente '{}' o '{}'?"
            case 'size':
                question_format = "¿Cuál es más grande '{}' o '{}'?"
            case 'mass':
                question_format = "¿Cuál es más pesado '{}' o '{}'?"
            case 'height':
                question_format = "¿Cuál es más alto '{}' o '{}'?"
            case 'hardness':
                question_format = "¿Cuál es más duro '{}' o '{}'?"
            case 'color':
                question_format = "¿Cuál es el color de '" + e['obj'] + "': '{}' o '{}'?"
            case 'material':
                question_format = "¿Cuál es el material de '" + e['obj'] + "': '{}' o '{}'?"
            case 'shape':
                question_format = "¿Cuál es la forma de '" + e['obj'] + "': '{}' o '{}'?"
            case _:
                raise ValueError("Wrong relation!")
        
        return make_question_and_answer(e, question_format)

    
    def write_ja(e):
        match e['relation']:
            case 'temperature':
                question_format = "{}と{}ではどちらが熱いでしょうか?"
            case 'size':
                question_format = "{}と{}ではどちらが大きいでしょうか?"
            case 'mass':
                question_format = "{}と{}ではどちらが重いでしょうか?"
            case 'height':
                question_format = "{}と{}ではどちらが高いでしょうか?"
            case 'hardness':
                question_format = "{}と{}ではどちらが固いでしょうか?"
            case 'color':
                question_format = "'" + e['obj'] + "'の色は何ですか: '{}'のか?'{}'のか?"
            case 'material':
                question_format = "'" + e['obj'] + "'の材質は何ですか: '{}'のか?'{}'のか?"
            case 'shape':
                question_format = "'" + e['obj'] + "'の形は何ですか: '{}'のか?'{}'のか?"
            case _:
                raise ValueError("Wrong relation!")

        return make_question_and_answer(e, question_format)

    def write_ru(e):
        match e['relation']:
            case 'temperature':
                question_format = "Что жарче: '{}' или '{}'?"
            case 'size':
                question_format = "Что больше: '{}' или '{}'?"
            case 'mass':
                question_format = "Что тяжелее: '{}' или '{}'?"
            case 'height':
                question_format = "Что выше: '{}' или '{}'?"
            case 'hardness':
                question_format = "Что жёстче: '{}' или '{}'?"
            case 'color':
                question_format = "Какого цвета'" + e['obj'] + "': '{}' или '{}'?"
            case 'material':
                question_format = "Из какого материала '" + e['obj'] + "': '{}' или '{}'?"
            case 'shape':
                question_format = "Какую форму имеет'" + e['obj'] + "': '{}' или '{}'?"
            case _:
                raise ValueError("Wrong relation!")

        return make_question_and_answer(e, question_format)

    lanFunctions = {
        'en': write_en,
        'ko': write_ko,
        'de': write_de,
        'zh-hans': write_zh,
        'es': write_es,
        'ja': write_ja,
        'ru': write_ru,
    }
    return lanFunctions[lan](e)



def create_dataset(lan):
    """
    Takes a concepts and returns a list of random few shots in english.

    :param str concept: The concept that is analyzed. 
                        Needs to be of the following: 'color', 'size', 'shape', 
                                                      'height', 'material', 'mass', 
                                                      'temperature', 'hardness' 
    """

    for c in ['color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness']:
        print(f"Creating Prompt DS for {lan}, concept {c}:")

        ds = load_dataset("nairdanus/multilingual_vec_dataset_" + lan, c)
        i = 0
        prompts = []
        for entry in ds["test"]:
            q, a = prompt_format(entry, lan)
            prompts.append({
                "id": i,
                "Q": q,
                "A": a,
            })
            i += 1
        prompt_ds = Dataset.from_list(prompts)
        prompt_ds.push_to_hub("nairdanus/VEC_prompt_" + lan, c, split="test")
        sleep(3)
        



if __name__ == "__main__":
    for lan in ["en"] + TGT_LANGS:
        create_dataset(lan)
