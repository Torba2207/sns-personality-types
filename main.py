import cohere

# Define API-Key
API_KEY = "1iiuRsRWNtyEZMG3YB2XeToINLiJXIq7QmvH4Zaw"

# Cohere client init
co = cohere.Client(API_KEY)

# Input Data
message = (
    "Hey there! I’m Alex, a creative mind who works as a graphic designer by day "
    "and dabbles in painting by night. I love exploring art galleries, finding hidden coffee shops, "
    "and attending live music events. I’m a big believer in enjoying life’s little moments—whether it’s a walk in "
    "the park or a cozy evening with a good movie. My friends say I’m loyal, thoughtful, and always ready with a terrible pun."
)

# Preamble for model
preamble = (
    "You are an AI trained to summarize information from dating app profile, such as hobby, education, lifestyle, "
    "age, gender, political orientation, etc. You need to give a response for example like that: Hobby: Music, "
    "Level of education: High, etc."
)

# Message to model
response = co.chat(
    message=message,
    preamble=preamble,
    model="command-r-08-2024",  # Chose model
    temperature=0.7,  # Grade of random
    max_tokens=200  # Response max length 
)

# Print response
print("Response:", response.text)
