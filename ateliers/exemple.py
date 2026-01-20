


# Mistral AI
api_key = "MISTRAL_API_KEY"
model = "mistral-small-2506"
client = Mistral(api_key=api_key)

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



