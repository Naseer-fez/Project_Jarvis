"""
core/agentic/neural_scheduler.py

A low-level task scheduler and prioritizer using a small neural network.
Runs entirely in NumPy at production runtime for zero startup overhead and low latency.
Contains a PyTorch neural network class used only for offline training.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import numpy as np

# Conditional PyTorch import for training mode only
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object  # Fallback for subclassing


class VocabularyVectorizer:
    """Super lightweight binary bag-of-words text vectorizer."""

    def __init__(self, vocabulary: dict[str, int] | None = None) -> None:
        self.vocabulary = vocabulary or {}

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary from a list of texts."""
        vocab = set()
        for text in texts:
            words = self._tokenize(text)
            vocab.update(words)
        
        # Sort for deterministic index assignment
        sorted_vocab = sorted(list(vocab))
        self.vocabulary = {word: idx for idx, word in enumerate(sorted_vocab)}

    def transform(self, text: str) -> np.ndarray:
        """Transform a text string into a binary bag-of-words NumPy vector."""
        vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        words = self._tokenize(text)
        for word in words:
            if word in self.vocabulary:
                vector[self.vocabulary[word]] = 1.0
        return vector

    def _tokenize(self, text: str) -> list[str]:
        """Clean and tokenize text into words."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in cleaned.split() if w]

    def to_dict(self) -> dict[str, Any]:
        return {"vocabulary": self.vocabulary}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VocabularyVectorizer:
        return cls(vocabulary=data.get("vocabulary"))


if HAS_TORCH:
    class NeuralSchedulerNet(nn.Module):
        """PyTorch neural network architecture used for offline training."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            # Priority classifier: 10 classes corresponding to priority levels 1 (highest) to 10 (lowest)
            self.priority_head = nn.Linear(32, 10)
            # Delay regressor: predicts log(delay_seconds + 1)
            self.delay_head = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            features = self.shared(x)
            priority_logits = self.priority_head(features)
            log_delay = self.delay_head(features)
            return priority_logits, log_delay
else:
    class NeuralSchedulerNet:  # type: ignore
        """Fallback when PyTorch is not available."""
        def __init__(self, *args, **kwargs) -> None:
            pass


class NeuralScheduler:
    """NumPy-based neural task scheduler for fast production inference."""

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = weights_path or os.path.join(
            os.path.dirname(__file__), "neural_scheduler_weights.json"
        )
        self.vectorizer = VocabularyVectorizer()
        self.weights: dict[str, np.ndarray] = {}
        self.loaded = False
        self.load_model()

    def load_model(self) -> bool:
        """Load model weights and vocabulary from JSON file."""
        if not os.path.exists(self.weights_path):
            return False
        
        try:
            with open(self.weights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Load vectorizer vocabulary
            self.vectorizer = VocabularyVectorizer.from_dict(data.get("vectorizer", {}))
            
            # Load NumPy weight arrays
            serialized_weights = data.get("weights", {})
            self.weights = {
                name: np.array(weights_list, dtype=np.float32)
                for name, weights_list in serialized_weights.items()
            }
            self.loaded = True
            return True
        except Exception:
            self.loaded = False
            return False

    def predict(self, text: str) -> tuple[int, float]:
        """
        Predict the task priority and execution delay.
        
        Returns:
            Tuple[int, float]: (priority_1_to_10, delay_seconds)
        """
        if not self.loaded:
            # Safe fallbacks (Priority 5, Delay 0s)
            return 5, 0.0

        try:
            # 1. Vectorize text
            x = self.vectorizer.transform(text)
            
            # 2. Forward pass through shared layers
            # Layer 1
            w1 = self.weights["fc1_weight"]
            b1 = self.weights["fc1_bias"]
            # PyTorch linear is: x @ w.T + b
            h1 = np.maximum(0.0, np.dot(x, w1.T) + b1)
            
            # Layer 2
            w2 = self.weights["fc2_weight"]
            b2 = self.weights["fc2_bias"]
            h2 = np.maximum(0.0, np.dot(h1, w2.T) + b2)
            
            # 3. Predict Priority (10-class Classification)
            wp = self.weights["priority_weight"]
            bp = self.weights["priority_bias"]
            priority_logits = np.dot(h2, wp.T) + bp
            priority_class = int(np.argmax(priority_logits))
            priority = priority_class + 1  # Map 0-9 index to 1-10 priority scale
            
            # 4. Predict Delay (log-delay continuous regression)
            wd = self.weights["delay_weight"]
            bd = self.weights["delay_bias"]
            log_delay = np.dot(h2, wd.T) + bd
            
            # Convert back from log scale: delay_seconds = exp(log_delay) - 1.0
            predicted_delay = float(np.expm1(log_delay[0]))
            
            # Safety checks and rounding
            if predicted_delay < 5.0:
                delay_seconds = 0.0
            else:
                delay_seconds = float(np.round(predicted_delay))
            
            return priority, delay_seconds
        except Exception:
            return 5, 0.0
