from datasets import load_dataset
import random


SEED = 42 # TODO: Only for testing.


def load_data(lan: str, concept: str):
    """
    Takes a chosen concept in the chosen language and returns a ds object.

    :param str lan: Language of the data. 
                    Possible Languages: 'en', 'ko', 'de', 'zh-CN', 'es', 'ja'
    :param str concept: The concept that is analyzed. 
                        Needs to be of the following: 'color', 'size', 'shape', 
                                                      'height', 'material', 'mass', 
                                                      'temperature', 'hardness' 
    """

    if lan not in ['en', 'ko', 'de', 'zh-CN', 'es', 'ja']:
        raise ValueError("Please use one of the following languages: 'en', 'ko', 'de', 'zh-CN', 'es' or 'ja'")

    if concept not in ['color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness']:
        raise ValueError("Please provide one of the following concepts: 'color', 'size', 'shape', 'height', 'material', 'mass', 'temperature' or 'hardness'")

    if lan == "en":
        ds = load_dataset("tobiaslee/VEC", concept)
        ds = ds['test'].map(lambda example: {'language': 'en'})
    else:
        ds = load_dataset("WindOcean/multilingual_vec_dataset", concept)
        ds = ds['test'].filter(lambda x: x['language'] == lan)

    return ds

def create_few_shots(concept: str, amount=0):
    """
    Takes a concepts and returns a list of random few shots in english.

    :param str concept: The concept that is analyzed. 
                        Needs to be of the following: 'color', 'size', 'shape', 
                                                      'height', 'material', 'mass', 
                                                      'temperature', 'hardness' 
    """
    ds = load_data(concept=concept, lan='en')
    if amount != 0:
        ds_samples = ds.shuffle(seed=SEED).select(range(amount))
    else:
        ds.shuffle(seed=SEED)
        ds_samples = ds


    def write_prompt(e):
        if "obj1" in ds_samples.column_names: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"Q: Is '{e['obj1']}' hotter than '{e['obj2']}'? A: {'Yes' if e['label'] else 'No'}"
                case 'size':
                    return f"Q: Is '{e['obj1']}' bigger than '{e['obj2']}'? A: {'Yes' if e['label'] else 'No'}"
                case 'mass':
                    return f"Q: Is '{e['obj1']}' heavier than '{e['obj2']}'? A: {'Yes' if e['label'] else 'No'}"
                case 'height':
                    return f"Q: Is '{e['obj1']}' taller than '{e['obj2']}'? A: {'Yes' if e['label'] else 'No'}"
                case 'hardness':
                    return f"Q: Is '{e['obj1']}' harder than '{e['obj2']}'? A: {'Yes' if e['label'] else 'No'}"
                case _:
                    raise ValueError("Wrong relation!")
        
        else: # color, material, shape
            if random.random() > 0.5:
                # return f"Q: Does {e['obj']} have the {e['relation']} '{e['positive']}' or '{e['negative']}'? A: {e['positive']}" 
                return f"Q: What is the {e['relation']} of {e['obj']}: '{e['positive']}' or '{e['negative']}'? A: {e['positive']}" 
            else:
                return f"Q: What is the {e['relation']} of {e['obj']}: '{e['negative']}' or '{e['positive']}'? A: {e['positive']}" 

    for entry in ds_samples:
        yield write_prompt(entry)


if __name__ == "__main__":
    for c in ['color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness']:
        for p in create_few_shots(c):
            with open("dataset/english_prompts_" + c + ".txt", "a") as f:
                f.write(p+"\n")



