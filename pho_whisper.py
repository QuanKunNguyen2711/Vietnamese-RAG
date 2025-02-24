from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large", device='cuda')