import os
import time
from brain.config import config
from benchmarks.runner import run_brain, load_suite

def sweep_darkness_decay():
    decays = [0.85, 0.90, 0.95]
    suite = load_suite()[:3] # fast sweep on first 3 turns
    
    print("Sweeping DARKNESS_DECAY...")
    for d in decays:
        print(f"--- Running with decay {d} ---")
        config.graph.darkness_decay = d
        run_brain(suite)

if __name__ == "__main__":
    sweep_darkness_decay()
