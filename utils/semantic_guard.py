# utils/semantic_guard.py
from typing import List, Tuple, Dict, Any
import numpy as np
from bert_score import BERTScorer

def _windows_from_text(text: str, window_words: int, stride_words: int, max_windows: int) -> List[str]:
    toks = text.split()
    n = len(toks)
    if n == 0:
        return []
    if n <= window_words:
        return [" ".join(toks)]
    wins, start, count = [], 0, 0
    while start < n and count < max_windows:
        end = min(n, start + window_words)
        wins.append(" ".join(toks[start:end]))
        start += stride_words
        count += 1
    return wins

class BertWindowGuard:
    """
    Sliding-window semantic guard using BERTScore.
    - Pre-loads the scorer so you don't reload a model for every query.
    - For each window, computes BERTScore against each template and takes the max F1.
    """
    def __init__(
        self,
        templates: List[str],
        model_type: str = "microsoft/deberta-base-mnli",  # fast/strong; change to roberta-large if you have GPU
        lang: str = "en",
        device: str = None,     # "cuda" or "cpu" (None lets BERTScore choose)
        batch_size: int = 16,
        window_words: int = 50,
        stride_words: int = 15,
        max_windows: int = 40,
        threshold_f1: float = 0.88,   # start around 0.85–0.90; tune on your dev set
        rescale_with_baseline: bool = True
    ):
        assert len(templates) > 0, "Provide at least one attack template"
        self.templates = templates
        self.scorer = BERTScorer(
            model_type=model_type,
            lang=lang,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
        )
        self.batch_size = batch_size
        self.window_words = window_words
        self.stride_words = stride_words
        self.max_windows = max_windows
        self.threshold_f1 = threshold_f1

    def score_windows(self, text: str) -> Dict[str, Any]:
        """Return the best F1 per window across templates + metadata."""
        wins = _windows_from_text(text, self.window_words, self.stride_words, self.max_windows)
        if not wins:
            return {"windows": [], "best_f1_per_window": [], "best_template_idx": [], "best_template": []}

        best_f1 = np.zeros(len(wins), dtype=np.float32)
        best_t_idx = np.full(len(wins), -1, dtype=np.int32)

        # For each template, score all windows at once (batch)
        for t_idx, template in enumerate(self.templates):
            # BERTScore API: lists must be same length; compare each window to the SAME template
            refs = [template] * len(wins)
            P, R, F1 = self.scorer.score(wins, refs, batch_size=self.batch_size)
            f1 = F1.detach().cpu().numpy().astype(np.float32)
            # keep the best template per window
            update_mask = f1 > best_f1
            best_f1[update_mask] = f1[update_mask]
            best_t_idx[update_mask] = t_idx

        best_templates = [self.templates[i] if i >= 0 else "" for i in best_t_idx.tolist()]
        return {
            "windows": wins,
            "best_f1_per_window": best_f1.tolist(),
            "best_template_idx": best_t_idx.tolist(),
            "best_template": best_templates,
        }

    def detect(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns:
          flagged (bool): True if any window ≥ threshold_f1
          info (dict): details (best window index, score, window text, matched template)
        """
        scores = self.score_windows(text)
        if not scores["windows"]:
            return False, {"reason": "empty_input"}

        f1s = np.array(scores["best_f1_per_window"], dtype=np.float32)
        k = int(np.argmax(f1s))
        max_f1 = float(f1s[k])
        if max_f1 >= self.threshold_f1:
            return True, {
                "reason": "semantic_match",
                "best_f1": max_f1,
                "best_window_idx": k,
                "best_window_text": scores["windows"][k],
                "matched_template": scores["best_template"][k],
            }
        return False, {"reason": "below_threshold", "max_f1": max_f1}
