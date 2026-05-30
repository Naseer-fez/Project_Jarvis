#!/usr/bin/env python
"""
bin/train_scheduler.py

Training script for the lightweight neural task scheduler.
Generates synthetic task/goal scheduling requests, trains a tiny neural network in PyTorch,
and serializes the vocabulary and weights to a JSON file for pure NumPy inference.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.agentic.neural_scheduler import VocabularyVectorizer, NeuralSchedulerNet, NeuralScheduler


def generate_dataset() -> list[tuple[str, int, float]]:
    """
    Generate a diverse synthetic dataset of task scheduling queries.
    Returns:
        List of tuples: (text, priority_1_to_10, delay_seconds)
    """
    dataset = []

    # Helper lists for generation
    urgent_actions = ["fix", "restart", "reboot", "restore", "recover", "stop", "abort", "kill", "rescue", "patch"]
    urgent_nouns = ["production server", "database", "auth system", "payment gateway", "checkout page", "login issue", "security breach", "memory leak", "crash", "network interface"]
    urgent_modifiers = ["now", "immediately", "asap", "stat", "urgently", "right away", "critical task", "emergency"]

    normal_actions = ["check", "verify", "ping", "test", "analyze", "inspect", "view", "query", "run scan on", "evaluate"]
    normal_nouns = ["server status", "log files", "cpu usage", "memory consumption", "api response", "disk space", "ssl certificate", "latest logs"]
    
    casual_actions = ["remind me to call", "email", "notify", "message", "ping", "slack", "schedule meeting with", "talk to"]
    casual_nouns = ["john", "sarah", "boss", "team", "client", "mom", "friend", "manager", "designer", "developer"]

    deferred_actions = ["clean", "archive", "delete", "organize", "sort", "backup", "optimize", "update", "refresh"]
    deferred_nouns = ["downloads folder", "temp directory", "old backups", "system logs", "workspace files", "profile settings", "config variables", "database indexes"]

    # 1. IMMEDIATE & CRITICAL (Priority 1-2, Delay 0s)
    # Generate around 150 examples
    for act in urgent_actions:
        for noun in urgent_nouns:
            mod = random.choice(urgent_modifiers)
            text = f"{act} {noun} {mod}"
            priority = random.choice([1, 2])
            dataset.append((text, priority, 0.0))

            # Variations
            text_alt = f"{mod}: {act} the {noun}"
            dataset.append((text_alt, priority, 0.0))

    # 2. SOON / SHORT TERM (Priority 3-5, Delay 300s - 1800s (5 - 30 minutes))
    # Generate around 150 examples
    short_times = [
        ("5 minutes", 300.0),
        ("5 mins", 300.0),
        ("10 minutes", 600.0),
        ("10 mins", 600.0),
        ("15 minutes", 900.0),
        ("20 minutes", 1200.0),
        ("30 minutes", 1800.0),
        ("half an hour", 1800.0)
    ]
    for act in normal_actions:
        for noun in normal_nouns:
            time_str, seconds = random.choice(short_times)
            prefix = random.choice(["remind me to", "set goal to", "schedule", "please", "don't forget to"])
            text = f"{prefix} {act} {noun} in {time_str}"
            priority = random.choice([3, 4, 5])
            dataset.append((text, priority, seconds))
            
            # Alt format
            text_alt = f"in {time_str} {act} the {noun}"
            dataset.append((text_alt, priority, seconds))

    # 3. MEDIUM TERM / DELAYED (Priority 5-7, Delay 3600s - 28800s (1 - 8 hours))
    medium_times = [
        ("1 hour", 3600.0),
        ("one hour", 3600.0),
        ("2 hours", 7200.0),
        ("two hrs", 7200.0),
        ("3 hours", 10800.0),
        ("4 hours", 14400.0),
        ("8 hours", 28800.0),
        ("tonight", 28800.0),
        ("this evening", 21600.0)
    ]
    for act in casual_actions:
        for noun in casual_nouns:
            time_str, seconds = random.choice(medium_times)
            prefix = random.choice(["remind me to", "set goal to", "schedule a reminder to", "i need to", "remember to"])
            text = f"{prefix} {act} {noun} in {time_str}"
            priority = random.choice([5, 6, 7])
            dataset.append((text, priority, seconds))

            # Alt format
            text_alt = f"{act} {noun} in {time_str}"
            dataset.append((text_alt, priority, seconds))

    # 4. LONG TERM / DEFERRED (Priority 7-10, Delay 86400s - 604800s (1 - 7 days))
    long_times = [
        ("tomorrow", 86400.0),
        ("tomorrow morning", 86400.0),
        ("next day", 86400.0),
        ("in 24 hours", 86400.0),
        ("next week", 604800.0),
        ("in a week", 604800.0),
        ("next monday", 604800.0),
        ("later", 86400.0)
    ]
    for act in deferred_actions:
        for noun in deferred_nouns:
            time_str, seconds = random.choice(long_times)
            prefix = random.choice(["remind me to", "schedule a task to", "when possible", "later, please", "i want to"])
            text = f"{prefix} {act} {noun} {time_str}"
            priority = random.choice([7, 8, 9, 10])
            dataset.append((text, priority, seconds))

            text_alt = f"{act} {noun} {time_str}"
            dataset.append((text_alt, priority, seconds))

    # Shuffle for training
    random.shuffle(dataset)
    return dataset


def train():
    print("Generating synthetic scheduling dataset...")
    dataset = generate_dataset()
    print(f"Generated {len(dataset)} training examples.")

    # Extract texts and targets
    texts = [item[0] for item in dataset]
    # Priority is 1-10 -> target class index is 0-9
    priority_targets = np.array([item[1] - 1 for item in dataset], dtype=np.int64)
    # Delay is continuous -> target log_delay is log(delay_seconds + 1.0)
    delay_targets = np.array([math.log1p(item[2]) for item in dataset], dtype=np.float32)

    # Fit vectorizer
    print("Fitting text vectorizer...")
    vectorizer = VocabularyVectorizer()
    vectorizer.fit(texts)
    vocab_size = len(vectorizer.vocabulary)
    print(f"Vocabulary size: {vocab_size} words.")

    # Vectorize inputs
    X_list = [vectorizer.transform(text) for text in texts]
    X_train = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_priority = torch.tensor(priority_targets, dtype=torch.long)
    y_delay = torch.tensor(delay_targets, dtype=torch.float32).unsqueeze(1)

    # Build model
    print("Initializing neural network...")
    model = NeuralSchedulerNet(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    criterion_priority = nn.CrossEntropyLoss()
    criterion_delay = nn.MSELoss()

    # Train model
    epochs = 200
    batch_size = 32
    num_samples = len(dataset)

    print(f"Training for {epochs} epochs...")
    model.train()
    for epoch in range(1, epochs + 1):
        # Mini-batch training
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        epoch_p_loss = 0.0
        epoch_d_loss = 0.0
        
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train[indices]
            batch_yp = y_priority[indices]
            batch_yd = y_delay[indices]

            optimizer.zero_grad()
            p_logits, d_pred = model(batch_x)

            loss_priority = criterion_priority(p_logits, batch_yp)
            loss_delay = criterion_delay(d_pred, batch_yd)
            
            # Loss weighting: balance classification and regression
            loss = loss_priority + 0.5 * loss_delay
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(indices)
            epoch_p_loss += loss_priority.item() * len(indices)
            epoch_d_loss += loss_delay.item() * len(indices)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} - Loss: {epoch_loss/num_samples:.4f} "
                  f"(Priority: {epoch_p_loss/num_samples:.4f}, Delay MSE: {epoch_d_loss/num_samples:.4f})")

    # Evaluate
    model.eval()
    with torch.no_grad():
        p_logits, d_pred = model(X_train)
        p_classes = torch.argmax(p_logits, dim=1)
        correct_p = (p_classes == y_priority).sum().item()
        p_acc = correct_p / num_samples
        
        mae_delay = torch.mean(torch.abs(torch.expm1(d_pred) - torch.expm1(y_delay))).item()
        print(f"\nTraining Evaluation:")
        print(f"- Priority Accuracy: {p_acc * 100:.2f}%")
        print(f"- Mean Absolute Error on Delay: {mae_delay:.2f} seconds")

    # Serialize weights for NumPy inference
    print("\nExtracting and serializing weights...")
    state_dict = model.state_dict()
    
    weights_data = {
        "vectorizer": vectorizer.to_dict(),
        "weights": {
            "fc1_weight": state_dict["shared.0.weight"].numpy().tolist(),
            "fc1_bias": state_dict["shared.0.bias"].numpy().tolist(),
            "fc2_weight": state_dict["shared.2.weight"].numpy().tolist(),
            "fc2_bias": state_dict["shared.2.bias"].numpy().tolist(),
            "priority_weight": state_dict["priority_head.weight"].numpy().tolist(),
            "priority_bias": state_dict["priority_head.bias"].numpy().tolist(),
            "delay_weight": state_dict["delay_head.weight"].numpy().tolist(),
            "delay_bias": state_dict["delay_head.bias"].numpy().tolist(),
        }
    }

    output_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "core", "agentic", "neural_scheduler_weights.json"
    ))
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2)
    print(f"Saved weights and vocabulary to {output_path}")

    # Validate NumPy inference output against PyTorch model
    print("Verifying NumPy inference implementation...")
    numpy_model = NeuralScheduler(weights_path=output_path)
    
    test_queries = [
        "restart the database immediately",
        "remind me to check server status in 5 mins",
        "remind me to message john in 2 hours",
        "clean up the old log files tomorrow morning",
    ]

    print("\nValidation Results on sample queries:")
    for query in test_queries:
        # PyTorch prediction
        x_vec = torch.tensor(vectorizer.transform(query), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_log, d_log = model(x_vec)
            py_p = int(torch.argmax(p_log, dim=1).item()) + 1
            py_d = float(torch.expm1(d_log[0][0]).item())
            if py_d < 5.0:
                py_d = 0.0
            else:
                py_d = round(py_d)
        
        # NumPy prediction
        np_p, np_d = numpy_model.predict(query)
        
        print(f"Query: '{query}'")
        print(f"  PyTorch: Priority={py_p}, Delay={py_d}s")
        print(f"  NumPy:   Priority={np_p}, Delay={np_d}s")
        assert np_p == py_p, f"Priority mismatch: {np_p} vs {py_p}"
        assert abs(np_d - py_d) < 1e-2, f"Delay mismatch: {np_d} vs {py_d}"
        
    print("\nNumPy vs PyTorch validation successful! Weights match exactly.")


if __name__ == "__main__":
    train()
