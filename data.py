from datasets import load_dataset


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

def create_few_shots(concept: str, amount: int):
    """
    Takes a concepts and returns a list of random few shots in english.

    :param str concept: The concept that is analyzed. 
                        Needs to be of the following: 'color', 'size', 'shape', 
                                                      'height', 'material', 'mass', 
                                                      'temperature', 'hardness' 
    """
    ds = load_data(concept=concept, lan='en')
    ds_samples = ds.shuffle(seed=SEED).select(range(amount))

    def write_prompt(e):
        if "obj1" in ds_samples.column_names: # Comparing data (obj1 v obj2)
            pass # TODO
        else:
            return f"{e['obj']} has the {e['relation']} {e['positive']}"

    for entry in ds_samples:
        yield write_prompt(entry)


if __name__ == "__main__":
    for p in create_few_shots("color", 3):
        print(p)



