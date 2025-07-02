class ModelConfig:
    def __init__(
        self,
        max_len: int = 3, 
        use_few_shots: bool = False, 
        no_repeat_ngram_size: int = 2, 
        top_k: int = 50, 
        top_p: float = 0.95, 
        temperature: float = 0.7,
        do_sample: bool = True
    ):

        self.max_len = max_len
        self.use_few_shots = use_few_shots
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.do_sample = do_sample

    def to_dict(self):
        return {
            "max_len": self.max_len,
            "use_few_shots": self.use_few_shots,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "do_sample": self.do_sample
        }

MODEL_CONFIG = ModelConfig(**{
        "max_len": 10, 
        "use_few_shots": False, 
        "no_repeat_ngram_size": 2, 
        "top_k": 50, 
        "top_p": 0.95, 
        "temperature": 0.7,
        "do_sample": True
    })

TGT_LANGS = ["es", "de", "ja", "ko", "zh-hans", "ru"]
SEED = 42