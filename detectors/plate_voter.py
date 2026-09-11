"""Consensus voting to stabilize noisy per-frame OCR reads."""

from collections import Counter, deque


class PlateVoter:
    def __init__(self, window=5, min_votes=3):
        self.window = window
        self.min_votes = min_votes
        self.buffer = deque(maxlen=window)

    def add(self, plate_text):
        if plate_text:
            self.buffer.append(plate_text)

    def get_consensus(self):
        if len(self.buffer) < self.min_votes:
            return None

        plate, votes = Counter(self.buffer).most_common(1)[0]
        if votes >= self.min_votes:
            return plate
        return None

    def reset(self):
        self.buffer.clear()
