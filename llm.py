from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from dotenv import load_dotenv
import os

load_dotenv()

class LLM_chain:
    def __init__(self, 
                 model_name: str = os.environ.get('LLM_MODEL'),
                 max_new_tokens: int = int(os.environ.get('MAX_NEW_TOKENS'))
                ):
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
        self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, 
                        low_cpu_mem_usage=True,
                        trust_remote_code=True, 
                        quantization_config=self._bnb_config,
                        device_map="auto"
                    )
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
    def get_llm_chain(self, temperature: float = 0.05, top_k: int = 50, top_p: float = 0.9, **kwargs):
        pipe = pipeline(
            task='text-generation',
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            model_kwargs={"use_cache": True}
            # **kwargs
        )
        
        return HuggingFacePipeline(pipeline=pipe, model_kwargs=kwargs)
    
    # def stream_generate(self, input_text: str, chunk_size: int = 50, **kwargs):
    #     # Tokenize input
    #     inputs = self._tokenizer(input_text, return_tensors="pt").to(self._model.device)
        
    #     # Generate the full response
    #     outputs = self._model.generate(
    #         **inputs,
    #         max_new_tokens=self.max_new_tokens,
    #         temperature=kwargs.get("temperature", 0.1),
    #         top_k=kwargs.get("top_k", 50),
    #         top_p=kwargs.get("top_p", 0.9),
    #         do_sample=True,
    #     )
        
    #     # Decode the response
    #     full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     yield full_text

    