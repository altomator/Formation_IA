import os
import sys
from mistralai import Mistral
import json
import csv

# data
output_resumes = './ocr_analysis'
output_decision = 'ocr_decision.csv'

prompt = "prompt_en.txt"


def call_mistral_api(prompt_content, file_path):
    print(f"...\033[7mcalling Mistral API for\033[m  {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        my_text = data.get("text", "")
        print("...processing text: ", my_text)
        chat_response = client.chat.complete(
            model = model,
            temperature = 0.0,
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
        return llm_answer

def compute_accuracy(result_file):
    with open(result_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip header
        correct_predictions = 0
        total_predictions = 0
        for row in reader:
            total_predictions += 1
            # Compare predicted classification with actual classification
            # row[2] is the predicted classification
            # row[0] is the actual classification if we suppose that the folder name is the actual classification
            if row[0] == row[2]:  
                correct_predictions += 1
                print(row[1], "==", row[0], " GREAT!")
            else:
                print("#",row[1], "should be '", row[0],"'#")
        print("---------------------\n", correct_predictions, "correct predictions out of", total_predictions, "predictions")
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nUsage: python extract_genre.py <input_folder> ")
    else:
        folder = sys.argv[1]

        # Mistral AI
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            print("Please set the MISTRAL_API_KEY environment variable!")
            sys.exit(1)
        model = "mistral-small-2506"
        #model = "mistral-large-2411"
        #model = "ministral-8b-2410"
        client = Mistral(api_key=api_key)

        # Read prompt file
        if not os.path.exists(prompt):
            print(f"# Error: The file '{prompt}' was not found! #")
            sys.exit(1)
        with open(prompt, 'r', encoding='utf-8') as prompt_file:
            prompt_content = prompt_file.read()
            print(f"...prompt is loaded.")

        # folders
        if not os.path.exists(os.path.join(output_resumes, folder)):
            print(f"...creating output folder in {output_resumes} for {folder}")
            os.makedirs(os.path.join(output_resumes, folder))
        # CSV file for decisions : init with header
        with open(output_decision, 'w', encoding='utf-8') as output_file:
            output_file.write("folder;file;classification\n")

        
        for txt_file in os.listdir(folder):
            print(f"\nNow processing file: {txt_file}...")
            if txt_file.endswith('.json'):
                llm_answer = call_mistral_api(prompt_content, os.path.join(folder, txt_file))
                print("\n...output:", llm_answer)
                output_file_path = os.path.join(output_resumes, folder,f"{os.path.splitext(txt_file)[0]}_resume.txt")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(llm_answer)
                print(f"...synthese saved to {output_file_path}")
                with open(output_decision, 'a', encoding='utf-8') as output_file:
                    decision = json.loads(llm_answer)
                    decision = decision.get("Decision")
                    print("\033[92mDecision:\033[m", decision)
                    output_file.write(sys.argv[1]+";"+txt_file + ";" + decision +"\n")
                print(f"...decision saved to {output_decision}")

        compute_accuracy(output_decision)
