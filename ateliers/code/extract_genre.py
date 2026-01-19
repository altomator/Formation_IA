import os
import sys
from mistralai import Mistral
from pydantic import BaseModel

import json


# data
output_resumes = './ocr_analysis'
prompt = "prompt_en.txt"

# Mistral AI
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("Please set the MISTRAL_API_KEY environment variable.")
    sys.exit(1)

client = Mistral(api_key=api_key)
model = "mistral-small-2506"
#model = "mistral-large-2411"
#model = "ministral-8b-2410"

class Rapport(BaseModel):
    Decision: str
    Reasoning: str
    Abstract: str



def call_mistral_api(prompt_content, file_path):
    print(f"...\033[7mcalling Mistral API for\033[m  {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        my_text = data.get("text", "")
        print("...processing text: ", my_text)
        chat_response = client.chat.complete(
            model = model,
            temperature = 0.0,
            #response_format = Rapport,
            messages = [
                {
                    "role": "system",
                    "content": f"{prompt_content}"
                },
                {
                    "role": "user",
                    "content": f"Process the provided text : {my_text}"
                },
            ],
            response_format={"type": "json_object"}

        )
        llm_answer = chat_response.choices[0].message.content
        print("...llm_answer:", llm_answer)
        try:
            extraction = Rapport.model_validate_json(llm_answer)
            print("\033[92mValidated:\033[m", extraction)
            return llm_answer
        except ValidationError as e:
            print(f"\033[91mValidation error: {e}\033[m")
            return None


# Mistral
client = Mistral(api_key=api_key)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nUsage: python extract_genre.py <input_folder> ")
    else:
        # Read prompt file
        if not os.path.exists(prompt):
            print(f"# Error: The file '{prompt}' was not found! #")
            sys.exit(1)
        print(f"...reading prompt from {prompt}...")
        with open(prompt, 'r', encoding='utf-8') as prompt_file:
            prompt_content = prompt_file.read()
        print(f"...prompt is loaded.")

        # folders
        if not os.path.exists(output_resumes):
            print(f"...creating output folder: {output_resumes}")
            os.makedirs(output_resumes)
    
        folder = sys.argv[1]
        for txt_file in os.listdir(folder):
            print(f"\nNow processing file: {txt_file}...")
            if txt_file.endswith('.txt'):
                output = call_mistral_api(prompt_content, os.path.join(folder, txt_file))
                print("\n...output:", output)
                output_file_path = os.path.join(output_resumes, f"{os.path.splitext(txt_file)[0]}_resume.txt")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(output)
                print(f"...synthese saved to {output_file_path}")
