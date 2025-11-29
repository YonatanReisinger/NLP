"""
ValenceDataset: Loads and manages valence score datasets.
"""
import csv
from typing import Dict, List


class ValenceDataset:
    """Handles loading word-valence data from CSV files."""

    def __init__(self, file_path: str):
        self.words: List[str] = []
        self.scores: List[float] = []
        self.word_to_score: Dict[str, float] = {}
        self._load(file_path)

    def _load(self, file_path: str) -> None:
        """Load word-valence pairs from a CSV file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    word = row[0].strip()
                    score = float(row[1])
                    self.words.append(word)
                    self.scores.append(score)
                    self.word_to_score[word] = score
