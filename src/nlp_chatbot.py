import openai

def chat_with_model(user_input):
    openai.api_key = 'your-api-key'
    
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=user_input,
      max_tokens=100
    )
    return response.choices[0].text.strip()
