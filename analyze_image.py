import json
from pathlib import Path

from ocr import SAMPLES_DIR, mean_confidence, ocr_image
from text_analysis import TextAnalyzer

RESULTS_DIR = Path(__file__).parent / "results"


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    analyzer = TextAnalyzer(top_n=5)
    images = sorted(
        p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )
    for path in images:
        text = ocr_image(path)
        stats = analyzer.analyze(text)
        record = {
            "file": path.name,
            "ocr_confidence": round(mean_confidence(path), 1),
            "text": text.strip(),
            **stats.to_dict(),
        }
        out_path = RESULTS_DIR / f"{path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"{path.name:<16}{stats.word_count:>5} words  {stats.language:<8}{stats.top_words[:3]}")
    print("Saved to", RESULTS_DIR)