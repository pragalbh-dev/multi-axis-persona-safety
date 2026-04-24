"""Dataset construction utilities.

Modules:
    build_dan_jailbreak: Download + preprocess the in-the-wild persona jailbreak set
        (Shen et al. 2024) into a 1,100-pair stratified sample mirroring Shah et al.'s
        schema. Primary persona-jailbreak eval dataset.
    reconstruct_shah_jailbreaks: Reconstruct Shah et al. (2311.03348) persona-modulation
        prompts via an LLM-as-attacker pipeline (Gemma 4 31B served by vLLM). Output
        schema matches the DAN sampled_1100 parquet so downstream eval code is
        dataset-agnostic.
"""
