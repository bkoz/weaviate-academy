#
# Weaviate Academy
# Course: 220 - Named Vectors
#
import weaviate
import weaviate.classes.config as wc
import pandas as pd
import requests
from datetime import datetime, timezone
import json
from weaviate.util import generate_uuid5
from tqdm import tqdm
import os
import zipfile
from pathlib import Path
import base64

# 
# Connect to the Weaviate
#
headers = {
    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
}

client = weaviate.connect_to_local(headers=headers)

# Check Weaviate status
try:
    assert client.is_live()

    # Create a movie collection.
    if client.collections.exists("MovieNVDemo"):
        print("Deleting existing MovieNVDemo collection.")
        client.collections.delete("MovieNVDemo")

    print("Creating new MovieNVDemo collection.")
    client.collections.create(
    name="MovieNVDemo",  # The name of the collection ('NV' for named vectors)
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
    vectorizer_config=[
        # Vectorize the movie title
        wc.Configure.NamedVectors.text2vec_openai(
            name="title", source_properties=["title"]
        ),
        # Vectorize the movie overview (summary)
        wc.Configure.NamedVectors.text2vec_openai(
            name="overview", source_properties=["overview"]
        ),
        # Vectorize the movie poster & title
        wc.Configure.NamedVectors.multi2vec_clip(
            name="poster_title",
            image_fields=[
                wc.Multi2VecField(name="poster", weight=0.9)
            ],  # 90% of the vector is from the poster
            text_fields=[
                wc.Multi2VecField(name="title", weight=0.1)
            ],  # 10% of the vector is from the title
        ),
    ],
    # Define the generative module
    generative_config=wc.Configure.Generative.openai(),
)
    data_url = "https://raw.githubusercontent.com/weaviate-tutorials/edu-datasets/main/movies_data_1990_2024.json"
    resp = requests.get(data_url)
    df = pd.DataFrame(resp.json())

    # Create a directory for the images
    img_dir = Path("scratch/imgs")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Download images
    posters_url = "https://raw.githubusercontent.com/weaviate-tutorials/edu-datasets/main/movies_data_1990_2024_posters.zip"
    posters_path = img_dir / "movies_data_1990_2024_posters.zip"
    posters_path.write_bytes(requests.get(posters_url).content)

    # Unzip the images
    with zipfile.ZipFile(posters_path, "r") as zip_ref:
        zip_ref.extractall(img_dir)

    # Get the collection
    movies = client.collections.get("MovieNVDemo")

    # Enter context manager
    with movies.batch.fixed_size(50) as batch:
        # Loop through the data
        for i, movie in tqdm(df.iterrows()):
            # Convert data types
            # Convert a JSON date to `datetime` and add time zone information
            release_date = datetime.strptime(movie["release_date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            # Convert a JSON array to a list of integers
            genre_ids = json.loads(movie["genre_ids"])
            # Convert image to base64
            img_path = img_dir / f"{movie['id']}_poster.jpg"
            with open(img_path, "rb") as file:
                poster_b64 = base64.b64encode(file.read()).decode("utf-8")

            # Build the object payload
            movie_obj = {
                "title": movie["title"],
                "overview": movie["overview"],
                "vote_average": movie["vote_average"],
                "genre_ids": genre_ids,
                "release_date": release_date,
                "tmdb_id": movie["id"],
                "poster": poster_b64,
            }

            # Add object to batch queue
            batch.add_object(
                properties=movie_obj,
                uuid=generate_uuid5(movie["id"]),
            )
            # Batcher automatically sends batches

    # Check for failed objects
    if len(movies.batch.failed_objects) > 0:
        print(f"Failed to import {len(movies.batch.failed_objects)} objects")
        for failed in movies.batch.failed_objects:
            print(f"e.g. Failed to import object with error: {failed.message}")
 
finally:
    print("Closing the Weaviate client connection.")
    client.close()