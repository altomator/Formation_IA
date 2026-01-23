import os
import csv
from PIL import Image
#pip install pillow
import json
import sys


ocr_output = "ocr_output"


# Mistral AI
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("Please set the MISTRAL_API_KEY environment variable.")
    sys.exit(1)
#model = "pixtral-12b-2409"
model = "pixtral-large-2411"

max_size = 3000

file_paths = []

          

def extract_text_from_image_with_mistral(image_path, ark, page, orientation):
    import base64
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    
    with open(image_path, "rb") as f:
        encoded_image = (
        "data:image/jpeg;base64,"
        + base64.b64encode(f.read()).decode('utf-8')
    )
    
    if orientation=="portrait":
        gallica_url = "https://openapi.bnf.fr/iiif/image/v3/ark:/12148/" + ark + "/f" + page + "/full/,"+ str(max_size) + "/0/default.jpg"
    else:
        gallica_url = "https://openapi.bnf.fr/iiif/image/v3/ark:/12148/" + ark + "/f" + page + "/full/" + str(max_size)+ ",/0/default.jpg"
    print(gallica_url)

    print("...calling Mistral", model)
    try:
        response = client.chat.complete(
            model=model,
            temperature=0.15,
            messages=[
                {"role": "system",
                 "content": "Return the answer in a JSON object with the next structure: "
                   "{\"text\": <text>"},
            {
            "role": "user",
            "content": "Extract the text printed on this image. If there is no text, return an empty string. "},
            {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": gallica_url,
            }
            ],
            "response_format": {
                "type": "json_object",
            }
        }
    ])
        ocr_json = response.choices[0].message.content
        return "\n".join(ocr_json.splitlines()[1:-1])

    except Exception as e:
        print(f"Error: {e}")
        return None


def get_iiif_info(ark, page):

    print("... calling IIIF info    ")
    iiif_url = "https://openapi.bnf.fr/iiif/image/v3/ark:/12148/" + ark + "/f" + page + "/info.json"
    #print(iiif_url)
    import requests
    try:
        with requests.get(iiif_url) as response:
            data = response.json()
            return data.get('width', 0), data.get('height', 0)
    except Exception as e:
        print(f"Error fetching IIIF info: {e}")
        return 0, 0 


def extract_ocr(input_dir, ocr_output):
    global file_paths

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    
    i = 1
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                print(f"Processing file {i}: ",file) 
                i=i+1
                
                file_path = os.path.join(root, file)
                
                tmp = file.split('-')
                ark = tmp[0]
                page = tmp[1]
                page = page.split('_')[1]
                output_file = os.path.join(ocr_output, f"{ark}_{page}.json")
                print("...writing in output_file: ", output_file)
                if os.path.exists(output_file):
                    print(f"...File already exists. Skipping.")
                    continue

                # call IIIF info
                (width, height) = get_iiif_info(ark, page)

                # Mistral
                if width > height:
                    orientation = "paysage"
                else:
                    orientation = "portrait"
                
                result = extract_text_from_image_with_mistral(file_path,ark,page,orientation)
                if result:
                    print(result)               
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(result + "\n")  
                else:
                    print("# No text found! #")
                
                         

if __name__ == "__main__":
    os.makedirs(ocr_output, exist_ok=True)

    if len(sys.argv) != 2:
        print("\nUsage: python extract_dim.py <input_folder> ")
    else:
        #print(extract_text_from_image_with_mistral("marie-claire_img/double_editorial/bpt6k4701101j-PAG_25_IL000001 - Grande.jpeg","paysage"))
        extract_ocr(sys.argv[1], ocr_output)

   
