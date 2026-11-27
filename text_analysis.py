import json
import re
from collections import Counter
from dataclasses import asdict, dataclass

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


if __name__ == "__main__":
    analyzer = TextAnalyzer(top_n=3)
    print(analyzer.words("Текст 2026, ТЕКСТ! м'ята"))

    print(analyzer.sentences("Перше речення. Друге!  Третє?"))
    print(analyzer.sentences("Один. Два"))
    print(analyzer.sentences("Один. Два. "))
    print(analyzer.top_words(analyzer.words("кіт і кіт і пес та кіт")))