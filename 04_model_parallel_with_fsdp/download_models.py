from transformers import AutoTokenizer, AutoModelForCausalLM

AutoTokenizer.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B", cache_dir=".cache")
AutoModelForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B", cache_dir=".cache")