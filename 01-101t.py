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

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),  
    auth_credentials=Auth.api_key(
        os.getenv("WEAVIATE_API_KEY")
    ),
    headers=headers,
)

# Check Weaviate status
try:
    assert client.is_live()

    # Retrieve the server meta information
    metainfo = client.get_meta()
    print(json.dumps(metainfo, indent=2))

    # Create a movie collection.
    if client.collections.exists("Movie"):
        print("Deleting existing Movie collection.")
        client.collections.delete("Movie")

    client.collections.create(
    name="Movie",
    properties=[
        wc.Property(name="title", data_type=wc.DataType.TEXT),
        wc.Property(name="overview", data_type=wc.DataType.TEXT),
        wc.Property(name="vote_average", data_type=wc.DataType.NUMBER),
        wc.Property(name="genre_ids", data_type=wc.DataType.INT_ARRAY),
        wc.Property(name="release_date", data_type=wc.DataType.DATE),
        wc.Property(name="tmdb_id", data_type=wc.DataType.INT),
    ],
    # Define the vectorizer module
    vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(),
    # Define the generative module
    generative_config=wc.Configure.Generative.openai()
    )

    # Get the collection
    movies = client.collections.get("Movie")

    # Enter context manager
    with movies.batch.fixed_size(batch_size=200) as batch:
        # Loop through the data
        for i, movie in tqdm(df.iterrows()):
            # Convert data types
            # Convert a JSON date to `datetime` and add time zone information
            release_date = datetime.strptime(movie["release_date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            # Convert a JSON array to a list of integers
            genre_ids = json.loads(movie["genre_ids"])

            # Build the object payload
            movie_obj = {
                "title": movie["title"],
                "overview": movie["overview"],
                "vote_average": movie["vote_average"],
                "genre_ids": genre_ids,
                "release_date": release_date,
                "tmdb_id": movie["id"],
            }

            # Add object to batch queue
            batch.add_object(
                properties=movie_obj,
                uuid=generate_uuid5(movie["id"])
                # references=reference_obj  # You can add references here
            )
            # Batcher automatically sends batches

    # Check for failed objects
    if len(movies.batch.failed_objects) > 0:
        print(f"Failed to import {len(movies.batch.failed_objects)} objects")
    else:
        print(f'{movies.batch.failed_objects = }')

    # Perform query
    print("Query = dystopian future")
    response = movies.query.near_text(
        query="dystopian future", limit=5, return_metadata=wq.MetadataQuery(distance=True)
    )

    # Inspect the response
    for o in response.objects:
        print(
            o.properties["title"], o.properties["release_date"].year
        )  # Print the title and release year (note the release date is a datetime object)
        print(
            f"Distance to query: {o.metadata.distance:.3f}\n"
        )  # Print the distance of the object from the query

    # Perform query
    print("BM25 query for history")
    response = movies.query.bm25(
        query="history", limit=5, return_metadata=wq.MetadataQuery(score=True)
    )

    # Inspect the response
    for o in response.objects:
        print(
            o.properties["title"], o.properties["release_date"].year
        )  # Print the title and release year (note the release date is a datetime object)
        print(
            f"BM25 score: {o.metadata.score:.3f}\n"
        )  # Print the BM25 score of the object from the query

    # Hybrid Query
    print("Hybrid Query")
    response = movies.query.hybrid(
        query="history", limit=5, return_metadata=wq.MetadataQuery(score=True)
    )

    # Inspect the response
    for o in response.objects:
        print(
            o.properties["title"], o.properties["release_date"].year
        )  # Print the title and release year (note the release date is a datetime object)
        print(
            f"Hybrid score: {o.metadata.score:.3f}\n"
        )  # Print the hybrid search score of the object from the query

    # Perform query
    print("Query using release_date filter")
    response = movies.query.near_text(
        query="dystopian future",
        limit=5,
        return_metadata=wq.MetadataQuery(distance=True),
        filters=wq.Filter.by_property("release_date").greater_than(datetime(2020, 1, 1))
    )

    # Inspect the response
    for o in response.objects:
        print(
            o.properties["title"], o.properties["release_date"].year
        )  # Print the title and release year (note the release date is a datetime object)
        print(
            f"Distance to query: {o.metadata.distance:.3f}\n"
        )  # Print the distance of the object from the query

    # Single Prompt
    print("Single prompt query: Translate this into French")

    response = movies.generate.near_text(
        query="dystopian future",
        limit=5,
        single_prompt="Translate this into French: {title}"
    )

    # Inspect the response
    for o in response.objects:
        print(o.properties["title"])  # Print the title
        print(o.generated)  # Print the generated text (the title, in French)

    # Generative Search
    print("Grouped task prompt query: What do these movies have in common?")
    response = movies.generate.near_text(
        query="dystopian future",
        limit=5,
        grouped_task="What do these movies have in common?",
        # grouped_properties=["title", "overview"]  # Optional parameter; for reducing prompt length
    )

    # Inspect the response
    for o in response.objects:
        print(o.properties["title"])  # Print the title
    print(response.generated)  # Print the generated text (the commonalities between them)



finally:
    print("Closing the Weaviate client connection.")
    client.close()