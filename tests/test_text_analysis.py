import pytest

from text_analysis import TextAnalyzer, load_json, save_json


@pytest.fixture
def analyzer():
    return TextAnalyzer(top_n=3)


def test_words_lowercase_and_no_digits(analyzer):
    assert analyzer.words("Текст 2026, ТЕКСТ!") == ["текст", "текст"]


def test_apostrophe_and_hyphen_keep_one_word(analyzer):
    assert analyzer.words("м'ята п’ять будь-який") == ["м'ята", "п’ять", "будь-який"]


def test_sentences(analyzer):
    assert analyzer.sentences("Перше речення. Друге! Третє?") == ["Перше речення", "Друге", "Третє"]


def test_top_words_skip_stop_words(analyzer):
    words = analyzer.words("кіт і кіт і пес та кіт")
    assert analyzer.top_words(words) == [("кіт", 3), ("пес", 1)]


def test_empty_text_does_not_crash(analyzer):
    stats = analyzer.analyze("")
    assert stats.word_count == 0
    assert stats.avg_word_length == 0
    assert stats.language == "unknown"


def test_analyze_count(analyzer):
    stats = analyzer.analyze("Кіт спить. Пес гавкає!\n")
    assert stats.word_count == 4
    assert stats.sentence_count == 2
    assert stats.char_count == 22
    assert stats.avg_word_length == 4.25


def test_json_round_trip(analyzer, tmp_path):
    stats = analyzer.analyze("Кіт спить. Кіт їсть.")
    path = tmp_path / "stats.json"
    save_json(stats, path)
    assert load_json(path) == stats


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Один.", 1),
        ("Один. Два. Три.", 3),
        ("Що це?! Не знаю...", 2),
        ("", 0),
    ],
)
def test_sentence_count(analyzer, text, expected):
    assert analyzer.analyze(text).sentence_count == expected