from transformers import AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig

# LLaMA model details
llama_model_name = "meta-llama/Llama-2-7b-hf"
llama_save_directory = "./checkpoints/llama"

# Download and save the LLaMA model
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)
llama_model.save_pretrained(llama_save_directory)

# Download and save the LLaMA tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_tokenizer.save_pretrained(llama_save_directory)

# Download and save the LLaMA config with model_type
llama_config = AutoConfig.from_pretrained(llama_model_name)
llama_config.model_type = "llama"
llama_config.save_pretrained(llama_save_directory)

# # DeBERTa model details
# deberta_model_name = "microsoft/deberta-base"
# deberta_save_directory = "./checkpoints/deberta"

# # Download and save the DeBERTa model
# deberta_model = AutoModelForQuestionAnswering.from_pretrained(deberta_model_name)
# deberta_model.save_pretrained(deberta_save_directory)

# # Download and save the DeBERTa tokenizer
# deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
# deberta_tokenizer.save_pretrained(deberta_save_directory)

# # Download and save the DeBERTa config with model_type
# deberta_config = AutoConfig.from_pretrained(deberta_model_name)
# deberta_config.model_type = "deberta"
# deberta_config.save_pretrained(deberta_save_directory)