import weaviate
import weaviate.classes.query as wq
import os
from datetime import datetime

def url_to_base64(url):
    import requests
    import base64

    image_response = requests.get(url)
    content = image_response.content
    return base64.b64encode(content).decode("utf-8")

# 
# Connect to Weaviate
#
headers = {
    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
}

client = weaviate.connect_to_local(headers=headers)

#     generative_config=wc.Configure.Generative.openai()

# Check Weaviate status
try:

    # Get the collection
    movies = client.collections.get("MovieMM")

    # Perform query
    src_img_path = "https://github.com/weaviate-tutorials/edu-datasets/blob/main/img/International_Space_Station_after_undocking_of_STS-132.jpg?raw=true"
    query_b64 = url_to_base64(src_img_path)

    response = movies.generate.near_image(
        near_image=query_b64,
        limit=5,
        single_prompt="Translate this into French: {title}"
    )

    # Inspect the response
    for o in response.objects:
        print(o.properties["title"])  # Print the title
        print(o.generated)  # Print the generated text (the title, in French)

finally:
    client.close()