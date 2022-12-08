# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

import transformers
import convert

transformers.models.gptj.modeling_gptj.GPTJBlock = convert.GPTJBlock

def download_model():
    
    # do a dry run of loading the huggingface model, which will download weights
    print("Model loading...")
    convert.GPTJForCausalLM.from_pretrained("yvan237/cedille-GPT-J-6B-8bit")
    print("done")
    
    print("Tokenizer loading...")
    transformers.AutoTokenizer.from_pretrained("yvan237/cedille-GPT-J-6B-8bit")
    print("done")

if __name__ == "__main__":
    download_model()
