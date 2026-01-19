import os
from pathlib import Path
from pydantic import BaseModel, Field

from mistralai import Mistral

class BrandExtraction(BaseModel):
    brand_name: str = Field(description="The primary brand name identified in the text")

def extract_brands_from_directory(directory_path: str, api_key: str):
    """
    Reads all .txt files in a directory and uses Mistral API to extract brand names
    validated by a Pydantic model.
    """
    client = Mistral(api_key=api_key)
    input_path = Path(directory_path)

    for file_path in input_path.glob("*.txt"):
        try:
            content = file_path.read_text(encoding="utf-8")
            
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {
                        "role": "system", 
                        "content": "Extract the brand name from the provided text. Return the result in JSON format."
                    },
                    {
                        "role": "user", 
                        "content": content
                    }
                ],
                response_format={"type": "json_object"}
            )

            # Parse and validate the JSON response using Pydantic
            raw_json = response.choices[0].message.content
            extraction = BrandExtraction.model_validate_json(raw_json)
            
            print(f"File: {file_path.name} -> Brand: {extraction.brand_name}")
            
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    # Ensure MISTRAL_API_KEY is set in your environment variables
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        extract_brands_from_directory("./ocr_test", api_key)
    else:
        print("Please set the MISTRAL_API_KEY environment variable.")
