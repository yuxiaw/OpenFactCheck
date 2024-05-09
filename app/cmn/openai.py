import openai

def validate_apikey(apikey):
    # Check if the API key is not None
    if apikey is None or apikey == "":
        return False
    
    # Check if the API key is valid
    client = openai.OpenAI(api_key=apikey)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True