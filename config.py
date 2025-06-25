class ModelConfig:
    def __init__(
        self,
        system_prompt: str = "",
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

    def to_dict(self):
        return {
            "max_len": self.max_len,
            "number_few_shots": self.number_few_shots,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "do_sample": self.do_sample
        }

MODEL_CONFIG = ModelConfig(**{
        "system_prompt": """
You are an expert in object recognition.
Using your expertise, always answer faithfully about the visual and embodied attributes of objects.
Always make sure to keep the answer the same way as in provided examples.
Answer in the same language in which the question was asked, unless explicitly instructed otherwise. If the input is in Spanish, respond in Spanish; if the input is in Korean, respond in Korean, etc.
Do not default to English unless the query is presented in English.
""",
        "max_len": 10, 
        "number_few_shots": 0, 
        "no_repeat_ngram_size": 2, 
        "top_k": 50, 
        "top_p": 0.95, 
        "temperature": 0.7,
        "do_sample": True
    })

TGT_LANGS = ["es", "de", "ja", "ko", "zh-hans", "ru"]
SEED = 42