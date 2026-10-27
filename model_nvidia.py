"""Configure the NVIDIA AI Endpoints models used by the evaluation benchmark.

Each item is a display name, model ID, and the .env variable containing that
model's NVIDIA API key. Edit this list to choose which models and keys to use.
"""

MODELS = [
    ("nvidia/nemotron-3.5-bolt-e4b", "nvidia/nemotron-3.5-bolt-e4b", "m1_api_key"),
    ("Mminimaxai/minimax-m3.1-flash", "minimaxai/minimax-m3.1-flash", "m2_api_key"),
]
