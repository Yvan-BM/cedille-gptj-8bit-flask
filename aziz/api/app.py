
import torch
from fastapi import Depends, FastAPI
from pydantic import BaseModel
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import convert


transformers.models.gptj.modeling_gptj.GPTJBlock = convert.GPTJBlock

app = FastAPI()

class InputRequest(BaseModel):
    input: str


class OuputResponse(BaseModel):
    output: str


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("yvan237/cedille-GPT-J-6B-8bit")
        self.model = convert.GPTJForCausalLM.from_pretrained("yvan237/cedille-GPT-J-6B-8bit")

        self.model.to(self.device)

    def predict(self, text):

        input_tokens = self.tokenizer(text, return_tensors='pt')
        input_tokens = {key: value.to(self.device) for key, value in input_tokens.items()}
        
        output = self.model.generate(**input_tokens, min_length=18, max_length=20, do_sample=True)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
    
        return output_text


model = Model()


def get_model():
    return model


@app.post("/text")
def predict(request: InputRequest, model: Model = Depends(get_model)):

    translate = model.predict(request.input)


    return {
        "output": translate
    }




