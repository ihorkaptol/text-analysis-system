import json
import re
from collections import Counter
from dataclasses import asdict, dataclass

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0
WORD_RE = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*")
SENTENCE_END_RE = re.compile(r"[.!?…]+")

STOP_WORDS = {
    # Ukrainian
    "і", "й", "та", "а", "але", "в", "у", "на", "з", "із", "зі", "до", "від",
    "за", "по", "про", "для", "що", "як", "це", "не", "так", "чи", "ж", "би",
    "він", "вона", "воно", "вони", "ми", "ви", "я", "ти", "його", "її", "їх",
    # English
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "be", "it", "this", "that", "as", "by",
    "from", "not",
}


@dataclass
class TextStats:
    char_count: int
    char_count_no_spaces: int
    word_count: int
    unique_word_count: int
    sentence_count: int
    avg_word_length: float
    top_words: list
    language: str

    def to_dict(self):
        return asdict(self)


class TextAnalyzer:
    def __init__(self, stop_words=None, top_n=10):
        self.stop_words = STOP_WORDS if stop_words is None else set(stop_words)
        self.top_n = top_n

    def words(self, text):
        return WORD_RE.findall(text.lower())

    def sentences(self, text):
        parts = SENTENCE_END_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    def top_words(self, words):
        meaningful = [w for w in words if w not in self.stop_words]
        return Counter(meaningful).most_common(self.top_n)

    def detect_language(self, text):
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"

    def analyze(self, text):
        text = text.strip()
        words = self.words(text)
        if words:
            avg = round(sum(len(w) for w in words) / len(words), 2)
        else:
            avg = 0.0
        return TextStats(
            char_count=len(text),
            char_count_no_spaces=len("".join(text.split())),
            word_count=len(words),
            unique_word_count=len(set(w for w in words)),
            sentence_count=len(self.sentences(text)),
            avg_word_length=avg,
            top_words=self.top_words(words),
            language=self.detect_language(text)
        )


def save_json(stats, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["top_words"] = [tuple(pair) for pair in data["top_words"]]
    return TextStats(**data)


if __name__ == "__main__":
    sample = (
        "Система аналізує друкований текст. Текст розпізнає Tesseract! "
        "Чи правильно система рахує слова? Ось м'ята і п’ять слів-прикладів…"
    )
    analyzer = TextAnalyzer(top_n=3)
    print(analyzer.analyze(sample))
