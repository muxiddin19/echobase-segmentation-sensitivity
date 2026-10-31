# Environment notes for TITAN Xp (Pascal, compute capability 6.1, 12 GB)

- No bf16, very slow fp16, no tensor cores; vLLM and FlashAttention need cc >= 7.0, so neither is usable.
- ASR: faster-whisper (CTranslate2) large-v3 with compute_type="int8" (int8 GPU kernels need cc >= 6.1, which this card has).
- LLM: llama.cpp (llama-cpp-python built with CUDA for sm_61), a 7-8B instruct model in Q4_K_M GGUF (about 5 GB), temperature 0, fixed seed.
- Diarization: pyannote/speaker-diarization-3.1. It needs a Hugging Face token and acceptance of the model's user conditions. Pin the version.
- PyTorch: VERIFY ON THE SERVER that the installed wheel still includes sm_61 kernels. Newer CUDA 12.8+ builds may have dropped Pascal; if so, pin an older cu121/cu118 wheel.
