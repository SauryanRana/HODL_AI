import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

class MCPModel:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", use_4bit=True, hf_token=None):
        """
        Load MCP LLM model for text generation.

        Args:
            model_name (str): Hugging Face model repo ID.
            use_4bit (bool): Load model with 4-bit quantization for efficiency.
            hf_token (str): Hugging Face token if needed for private models.
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token

        self.tokenizer = None
        self.model = None
        self.generator = None

        self._load_model()

    def _load_model(self):
        tokenizer_kwargs = {}
        model_kwargs = {}

        if self.hf_token:
            tokenizer_kwargs["use_auth_token"] = self.hf_token
            model_kwargs["use_auth_token"] = self.hf_token

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=False,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["device_map"] = None
            model_kwargs["torch_dtype"] = torch.float32

        print(f"Loading tokenizer for {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        print("Tokenizer loaded.")

        print(f"Loading model {self.model_name} ...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        print("Model loaded.")

        print("Creating text generation pipeline ...")
        # When using accelerate (device_map='auto'), do NOT pass device argument to pipeline
        if "device_map" in model_kwargs and model_kwargs["device_map"] == "auto":
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=True,
            )
        else:
            device_id = 0 if self.device == "cuda" else -1
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
                return_full_text=True,
            )
        print("Pipeline ready.")

    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_k=50, do_sample=True):
        """
        Generate text given a prompt.

        Args:
            prompt (str): The input prompt.
            max_new_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter.
            do_sample (bool): Whether to sample or do greedy decoding.

        Returns:
            str: The generated text after the prompt.
        """
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()

if __name__ == "__main__":
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    mcp = MCPModel(hf_token=hf_token)
    prompt = "Explain the $HODL token ecosystem in simple terms."
    print("Prompt:", prompt)
    print("Output:", mcp.generate(prompt))
