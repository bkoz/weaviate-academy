#
# Weaviate Academy
# Course: 101T - Working with text data
#
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes.config as wc
import weaviate.classes.query as wq
import pandas as pd
import requests
from datetime import datetime, timezone
from weaviate.util import generate_uuid5
from tqdm import tqdm
import os
import json

# Grab the movie data.
data_url = "https://raw.githubusercontent.com/weaviate-tutorials/edu-datasets/main/movies_data_1990_2024.json"
resp = requests.get(data_url)
df = pd.DataFrame(resp.json())

# 
# Connect to the Weaviate Cloud Instance
#
headers = {
    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
}

client = weaviate.connect_to_local()

# Check Weaviate status
try:
    assert client.is_live()

    # Retrieve the server meta information
    metainfo = client.get_meta()
    print(json.dumps(metainfo, indent=2))

    # Create a movie collection.
    if client.collections.exists("MovieMM"):
        print("Deleting existing MovieMM collection.")
        client.collections.delete("MovieMM")

    client.collections.create(
        name="MovieMM",  # The name of the collection ('MM' for multimodal)
        properties=[
            wc.Property(name="title", data_type=wc.DataType.TEXT),
            wc.Property(name="overview", data_type=wc.DataType.TEXT),
            wc.Property(name="vote_average", data_type=wc.DataType.NUMBER),
            wc.Property(name="genre_ids", data_type=wc.DataType.INT_ARRAY),
            wc.Property(name="release_date", data_type=wc.DataType.DATE),
            wc.Property(name="tmdb_id", data_type=wc.DataType.INT),
            wc.Property(name="poster", data_type=wc.DataType.BLOB),
        ],
        # Define & configure the vectorizer module
        vectorizer_config=wc.Configure.Vectorizer.multi2vec_clip(
            image_fields=[wc.Multi2VecField(name="poster", weight=0.9)],    # 90% of the vector is from the poster
            text_fields=[wc.Multi2VecField(name="title", weight=0.1)],      # 10% of the vector is from the title
        ),
        # Define the generative module
        generative_config=wc.Configure.Generative.openai()
    )

finally:
    print("Closing the Weaviate client connection.")
    client.close()