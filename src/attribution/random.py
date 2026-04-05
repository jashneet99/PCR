import random
from typing import Dict, List

class RandomIWF:
    def __init__(
        self,
        dataset: str,
        seed: int = 42
    ):
        self.dataset = dataset
        self.seed = seed

        self.punctuations = {",", ".", "?", "!"}
        self.field_map = {
            "esnli": ["premise", "hypothesis"],
            "comve": ["sentence0", "sentence1"],
            "ecqa": ["question"]
        }

    def clean_and_tokenize(self, text: str) -> List[str]:
        words = text.split()
        cleaned = []
        for w in words:
            if w and w[-1] in self.punctuations:
                w = w[:-1]
            cleaned.append(w.lower())
        return cleaned

    def __call__(self, item: Dict) -> Dict:
        item['iw_rand_feedback'] = {}
        
        word_set = set()
        for field in self.field_map[self.dataset]:
            text = item[field]
            words = self.clean_and_tokenize(text)
            word_set.update(words)

        word_list = list(word_set)

        random.seed(self.seed)
        random.shuffle(word_list)

        item['iw_rand_feedback']['final'] = word_list

        return item
