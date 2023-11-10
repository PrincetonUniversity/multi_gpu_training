from transformers import AutoTokenizer, AutoModelForCausalLM

AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", cache_dir=".cache")
AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", cache_dir=".cache")