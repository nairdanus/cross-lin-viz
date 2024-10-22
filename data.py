from datasets import load_dataset
import random
import os

SEED = 42 # TODO: Only for testing.


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


def prompt_format(ds_samples, e, lan):
    def write_en(e):
        if "obj1" in ds_samples.column_names: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"Q: Is '{e['obj1'].lower()}' hotter than '{e['obj2'].lower()}'? A: {'Yes' if e['label'] else 'No'}"
                case 'size':
                    return f"Q: Is '{e['obj1'].lower()}' bigger than '{e['obj2'].lower()}'? A: {'Yes' if e['label'] else 'No'}"
                case 'mass':
                    return f"Q: Is '{e['obj1'].lower()}' heavier than '{e['obj2'].lower()}'? A: {'Yes' if e['label'] else 'No'}"
                case 'height':
                    return f"Q: Is '{e['obj1'].lower()}' taller than '{e['obj2'].lower()}'? A: {'Yes' if e['label'] else 'No'}"
                case 'hardness':
                    return f"Q: Is '{e['obj1'].lower()}' harder than '{e['obj2'].lower()}'? A: {'Yes' if e['label'] else 'No'}"
                case _:
                    raise ValueError("Wrong relation!")
        
        else: # color, material, shape
            if random.random() > 0.5:
                return f"Q: What is the {e['relation']} of {e['obj']}: '{e['positive'].lower()}' or '{e['negative'].lower()}'? A: {e['positive'].lower()}" 
            else:
                return f"Q: What is the {e['relation']} of {e['obj']}: '{e['negative'].lower()}' or '{e['positive'].lower()}'? A: {e['positive'].lower()}" 
    
    def write_ko(e):
        if "obj1" in ds_samples.column_names: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"Q: '{e['obj1']}'가 '{e['obj2']}'보다 더 뜨겁나요? A: {'예' if e['label'] else '아니요'}"
                case 'size':
                    return f"Q: '{e['obj1']}'가 '{e['obj2']}'보다 큰가요? A: {'예' if e['label'] else '아니요'}"
                case 'mass':
                    return f"Q: '{e['obj1']}'가 '{e['obj2']}'보다 무겁나요? A: {'예' if e['label'] else '아니요'}"
                case 'height':
                    return f"Q: '{e['obj1']}'가 '{e['obj2']}'보다 키가 큰가요? A: {'예' if e['label'] else '아니요'}"
                case 'hardness':
                    return f"Q: '{e['obj1']}'가 '{e['obj2']}'보다 더 무겁나요? A: {'예' if e['label'] else '아니요'}"
                case _:
                    raise ValueError("Wrong relation!")
                
        else: # color, material, shape
            if random.random() > 0.5:
                return f"Q: {e['obj']}의 {translate_relation(e['relation'], lan)}은 무엇인가요: '{e['positive']}' 또는 '{e['negative']}'? A: {e['positive']}" 
            else:
                return f"Q: {e['obj']}의 {translate_relation(e['relation'], 'ko')}은 무엇인가요: '{e['negative']}' 또는 '{e['positive']}'? A: {e['positive']}" 
    
    def write_de(e):
        if "obj1" in ds_samples.column_names: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"Q: Ist '{e['obj1']}' heißer als '{e['obj2']}'? A: {'Ja' if e['label'] else 'Nein'}"
                case 'size':
                    return f"Q: Ist '{e['obj1']}' größer als '{e['obj2']}'? A: {'Ja' if e['label'] else 'Nein'}"
                case 'mass':
                    return f"Q: Ist '{e['obj1']}' schwerer als '{e['obj2']}'? A: {'Ja' if e['label'] else 'Nein'}"
                case 'height':
                    return f"Q: Ist '{e['obj1']}' höher als '{e['obj2']}'? A: {'Ja' if e['label'] else 'Nein'}"
                case 'hardness':
                    return f"Q: Ist '{e['obj1']}' härter als '{e['obj2']}'? A: {'Ja' if e['label'] else 'Nein'}"
                case _:
                    raise ValueError("Wrong relation!")
        
        else: # color, material, shape
            if random.random() > 0.5:
                if e['relation'] == 'material':
                    return f"Q: Aus welchem Material besteht {e['obj']}: '{e['positive'].lower()}' oder '{e['negative'].lower()}'? A: {e['positive'].lower()}"
                else:
                    return f"Q: Welche {translate_relation(e['relation'], 'de')} hat {e['obj']}: '{e['positive'].lower()}' oder '{e['negative'].lower()}'? A: {e['positive'].lower()}" 
            else:
                if e['relation'] == 'material':
                    return f"Q: Aus welchem Material besteht {e['obj']}: '{e['negative'].lower()}' oder '{e['positive'].lower()}'? A: {e['positive'].lower()}"
                else:
                    return f"Q: Welche {translate_relation(e['relation'], 'de')} hat {e['obj']}: '{e['negative'].lower()}' oder '{e['positive'].lower()}'? A: {e['positive'].lower()}" 
              
    def write_zh(e): #TODO
        if "obj1" in ds_samples.column_names: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"Q: '{e['obj1']}'比'{e['obj2']}'热吗? A: {'是' if e['label'] else '否'}"
                case 'size':
                    return f"Q: '{e['obj1']}'比'{e['obj2']}'大吗? A: {'是' if e['label'] else '否'}"
                case 'mass':
                    return f"Q: '{e['obj1']}'比'{e['obj2']}'重吗? A: {'是' if e['label'] else '否'}"
                case 'height':
                    return f"Q: '{e['obj1']}'比'{e['obj2']}'高吗? A: {'是' if e['label'] else '否'}"
                case 'hardness':
                    return f"Q: '{e['obj1']}'比'{e['obj2']}'硬吗? A: {'是' if e['label'] else '否'}"
                case _:
                    raise ValueError("Wrong relation!")
        
        else: # color, material, shape
            if random.random() > 0.5:
                return f"Q: {e['obj']}是什么{translate_relation(e['relation'], 'zh')}: 是'{e['positive']}'还是'{e['negative']}'? A: {e['positive']}" 
            else:
                return f"Q: {e['obj']}是什么{translate_relation(e['relation'], 'zh')}: 是'{e['negative']}'还是'{e['positive']}'? A: {e['positive']}" 
    
    def write_es(e):#TODO
        if "obj1" in ds_samples.column_names: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"Q: ¿Es '{e['obj1'].lower()}' más caliente que '{e['obj2'].lower()}'? A: {'Sí' if e['label'] else 'No'}"
                case 'size':
                    return f"Q: ¿Es '{e['obj1'].lower()}' más grande que '{e['obj2'].lower()}'? A: {'Sí' if e['label'] else 'No'}"
                case 'mass':
                    return f"Q: ¿Es '{e['obj1'].lower()}' más pesado que '{e['obj2'].lower()}'? A: {'Sí' if e['label'] else 'No'}"
                case 'height':
                    return f"Q: ¿Es '{e['obj1'].lower()}' más alto que '{e['obj2'].lower()}'? A: {'Sí' if e['label'] else 'No'}"
                case 'hardness':
                    return f"Q: ¿Es '{e['obj1'].lower()}' más duro que '{e['obj2'].lower()}'? A: {'Sí' if e['label'] else 'No'}"
                case _:
                    raise ValueError("Wrong relation!")
        
        else: # color, material, shape
            if random.random() > 0.5: 
                return f"Q: ¿Cuál es {translate_relation(e['relation'], 'es')} de {e['obj']}: '{e['positive'].lower()}' o '{e['negative'].lower()}'? A: {e['positive'].lower()}" 
            else:
                return f"Q: ¿Cuál es {translate_relation(e['relation'], 'es')} de {e['obj']}: '{e['negative'].lower()}' o '{e['positive'].lower()}'? A: {e['positive'].lower()}" 
    
    def write_ja(e):
        if "obj1" in ds_samples.column_names: # Comparing data (obj1 v obj2)
            match e['relation']:
                case 'temperature':
                    return f"Q: '{e['obj1']}'は'{e['obj2']}'より熱いか? A: {'はい' if e['label'] else 'いいえ'}"
                case 'size':
                    return f"Q: '{e['obj1']}'は'{e['obj2']}'より大きいか? A: {'はい' if e['label'] else 'いいえ'}"
                case 'mass':
                    return f"Q: '{e['obj1']}'は'{e['obj2']}'より重いか? A: {'はい' if e['label'] else 'いいえ'}"
                case 'height':
                    return f"Q: '{e['obj1']}'は'{e['obj2']}'より高いか? A: {'はい' if e['label'] else 'いいえ'}"
                case 'hardness':
                    return f"Q: '{e['obj1']}'は'{e['obj2']}'より固いか? A: {'はい' if e['label'] else 'いいえ'}"
                case _:
                    raise ValueError("Wrong relation!")
        
        else: # color 色, material材質, shape 形
            if random.random() > 0.5:
                return f"Q: {e['obj']}の{translate_relation(e['relation'], 'ja')}は何ですか: '{e['positive']}'のか?'{e['negative']}'のか? A: {e['positive']}です" 
            else:
                return f"Q: {e['obj']}の{translate_relation(e['relation'], 'ja')}は何ですか: '{e['negative']}'のか?'{e['positive']}'のか? A: {e['positive']}です" 
    


    lanFunctions = {
        'en': write_en,
        'ko': write_ko,
        'de': write_de,
        'zh-CN': write_zh,
        'es': write_es,
        'ja': write_ja,
    }
    return lanFunctions[lan](e)



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



def create_dataset(lan):
    """
    Takes a concepts and returns a list of random few shots in english.

    :param str concept: The concept that is analyzed. 
                        Needs to be of the following: 'color', 'size', 'shape', 
                                                      'height', 'material', 'mass', 
                                                      'temperature', 'hardness' 
    """
    dir_name = os.path.join('dataset', lan)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for c in ['color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness']:
        file_name = os.path.join(dir_name,c + ".csv")
        if os.path.exists(file_name):
            raise FileExistsError("Hallo, das gibt es schon!")
        
        ds = load_data(concept=c, lan=lan)

        i = 0

        for entry in ds:
            prompt = prompt_format(ds, entry, lan)
            with open(file_name, 'a') as f:
                f.write(str(i)+","+prompt+"\n")
            i += 1


def create_en():
    for c in ['color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness']:
        for i, p in enumerate(create_few_shots(c)):
            file_name = "dataset/prompts_en_" + c + ".csv"
            if os.path.exists(file_name):
                raise FileExistsError("Hallo, das gibt es schon!")
            with open(file_name, "a") as f:
                f.write(str(i)+","+p+"\n")



if __name__ == "__main__":
    for lan in ['en', 'ko', 'de', 'zh-CN', 'es', 'ja']:
        create_dataset(lan)
