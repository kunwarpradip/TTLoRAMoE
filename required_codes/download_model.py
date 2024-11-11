from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


def download_deberta_model():
    # DeBERTa model details
    model_name = "microsoft/deberta-base"
    save_directory = "./checkpoints/deberta"

    # Download and save the DeBERTa model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.save_pretrained(save_directory)

    # Download and save the DeBERTa tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)

    # Download and save the DeBERTa config with model_type
    config = AutoConfig.from_pretrained(model_name)
    config.model_type = "deberta"
    config.save_pretrained(save_directory)

    print(f"DeBERTa model and tokenizer saved to {save_directory}")

def download_roberta_model():
    # RoBERTa model details
    model_name = "roberta-base"
    save_directory = "./checkpoints/roberta"

    # Download and save the RoBERTa model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(save_directory)

    # Download and save the RoBERTa tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)

    # Download and save the RoBERTa config with model_type
    config = AutoConfig.from_pretrained(model_name)
    config.model_type = "roberta"
    config.save_pretrained(save_directory)

    print(f"RoBERTa model and tokenizer saved to {save_directory}")


if __name__ == "__main__":
    # download_deberta_model()
    download_roberta_model()