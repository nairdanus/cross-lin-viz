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
                question_format = "Was ist heißer: '{}' or '{}'?"
            case 'size':
                question_format = "Was ist größer: '{}' or '{}'?"
            case 'mass':
                question_format = "Was ist schwerer: '{}' or '{}'?"
            case 'height':
                question_format = "Was ist höher: '{}' or '{}'?"
            case 'hardness':
                question_format = "Was ist härter: '{}' or '{}'?"
            case 'color':
                question_format = "Welche Farbe hat '" + e['obj'] + "': '{}' or '{}'?"
            case 'material':
                question_format = "Aus welchem Material besteht '" + e['obj'] + "': '{}' or '{}'?"
            case 'shape':
                question_format = "Welche Form hat '" + e['obj'] + "': '{}' or '{}'?"
            case _:
                raise ValueError("Wrong relation!")

        return make_question_and_answer(e, question_format)


    def write_zh(e): #TODO
        if "obj1" in e: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"'{e['obj1']}'比'{e['obj2']}'热吗?", f"{'是' if e['label'] else '否'}"
                case 'size':
                    return f"'{e['obj1']}'比'{e['obj2']}'大吗?", f"{'是' if e['label'] else '否'}"
                case 'mass':
                    return f"'{e['obj1']}'比'{e['obj2']}'重吗?", f"{'是' if e['label'] else '否'}"
                case 'height':
                    return f"'{e['obj1']}'比'{e['obj2']}'高吗?", f"{'是' if e['label'] else '否'}"
                case 'hardness':
                    return f"'{e['obj1']}'比'{e['obj2']}'硬吗?", f"{'是' if e['label'] else '否'}"
                case _:
                    raise ValueError("Wrong relation!")
        
        else: # color, material, shape
            if random.random() > 0.5:
                return f"{e['obj']}是什么{translate_relation(e['relation'], 'zh')}: 是'{e['positive']}'还是'{e['negative']}'?", f"{e['positive']}" 
            else:
                return f"{e['obj']}是什么{translate_relation(e['relation'], 'zh')}: 是'{e['negative']}'还是'{e['positive']}'?", f"{e['positive']}" 
    
    def write_es(e): # TODO
        if "obj1" in e: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"¿Es '{e['obj1'].lower()}' más caliente que '{e['obj2'].lower()}'?", f"{'Sí' if e['label'] else 'No'}"
                case 'size':
                    return f"¿Es '{e['obj1'].lower()}' más grande que '{e['obj2'].lower()}'?", f"{'Sí' if e['label'] else 'No'}"
                case 'mass':
                    return f"¿Es '{e['obj1'].lower()}' más pesado que '{e['obj2'].lower()}'?", f"{'Sí' if e['label'] else 'No'}"
                case 'height':
                    return f"¿Es '{e['obj1'].lower()}' más alto que '{e['obj2'].lower()}'?", f"{'Sí' if e['label'] else 'No'}"
                case 'hardness':
                    return f"¿Es '{e['obj1'].lower()}' más duro que '{e['obj2'].lower()}'?", f"{'Sí' if e['label'] else 'No'}"
                case _:
                    raise ValueError("Wrong relation!")
        
        else: # color, material, shape
            if random.random() > 0.5: 
                return f"¿Cuál es {translate_relation(e['relation'], 'es')} de {e['obj']}: '{e['positive'].lower()}' o '{e['negative'].lower()}'?", f"{e['positive'].lower()}" 
            else:
                return f"¿Cuál es {translate_relation(e['relation'], 'es')} de {e['obj']}: '{e['negative'].lower()}' o '{e['positive'].lower()}'?", f"{e['positive'].lower()}" 
    
    def write_ja(e): # TODO
        if "obj1" in e: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"'{e['obj1']}'は'{e['obj2']}'より熱いか?", f"{'はい' if e['label'] else 'いいえ'}"
                case 'size':
                    return f"'{e['obj1']}'は'{e['obj2']}'より大きいか?", f"{'はい' if e['label'] else 'いいえ'}"
                case 'mass':
                    return f"'{e['obj1']}'は'{e['obj2']}'より重いか?", f"{'はい' if e['label'] else 'いいえ'}"
                case 'height':
                    return f"'{e['obj1']}'は'{e['obj2']}'より高いか?", f"{'はい' if e['label'] else 'いいえ'}"
                case 'hardness':
                    return f"'{e['obj1']}'は'{e['obj2']}'より固いか?", f"{'はい' if e['label'] else 'いいえ'}"
                case _:
                    raise ValueError("Wrong relation!")
        
        else: # color 色, material材質, shape 形
            if random.random() > 0.5:
                return f"{e['obj']}の{translate_relation(e['relation'], 'ja')}は何ですか: '{e['positive']}'のか?'{e['negative']}'のか?", f"{e['positive']}です" 
            else:
                return f"{e['obj']}の{translate_relation(e['relation'], 'ja')}は何ですか: '{e['negative']}'のか?'{e['positive']}'のか?", f"{e['positive']}です" 

    def write_ru(e): # TODO
        raise NotImplementedError(f"No prompt template for Russian!")

        match e['relation']:
            case 'temperature':
                question_format = "Was ist heißer: '{}' or '{}'?"
            case 'size':
                question_format = "Was ist größer: '{}' or '{}'?"
            case 'mass':
                question_format = "Was ist schwerer: '{}' or '{}'?"
            case 'height':
                question_format = "Was ist höher: '{}' or '{}'?"
            case 'hardness':
                question_format = "Was ist härter: '{}' or '{}'?"
            case 'color':
                question_format = "Welche Farbe hat '" + e['obj'] + "': '{}' or '{}'?"
            case 'material':
                question_format = "Aus welchem Material besteht '" + e['obj'] + "': '{}' or '{}'?"
            case 'shape':
                question_format = "Welche Form hat '" + e['obj'] + "': '{}' or '{}'?"
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
