#
# Weaviate Academy
# Course: 101V - BYO vectors
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
import cohere
from cohere import Client as CohereClient
from typing import List

# Grab the movie data.
# data_url = "https://raw.githubusercontent.com/weaviate-tutorials/edu-datasets/main/movies_data_1990_2024.json"
# resp = requests.get(data_url)
# df = pd.DataFrame(resp.json())

co_token = os.getenv("COHERE_APIKEY")
co = cohere.Client(co_token)

# 
# Connect to the Weaviate Cloud Instance
#
headers = {
    "X-Cohere-Api-Key": os.getenv("COHERE_API_KEY")
}

# client = weaviate.connect_to_weaviate_cloud(
#     cluster_url=os.getenv("WEAVIATE_URL"),  
#     auth_credentials=Auth.api_key(
#         os.getenv("WEAVIATE_API_KEY")
#     ),
#     headers=headers,
# )

# Connect to the local Weaviate instance using the header defined above.
client = weaviate.connect_to_local(
    headers=headers,
)

# Define a function to call the endpoint and obtain embeddings
def vectorize(cohere_client: CohereClient, texts: List[str]) -> List[List[float]]:

    response = cohere_client.embed(
        texts=texts, model="embed-multilingual-v3.0", input_type="search_document"
    )

    return response.embeddings

# Get the source data
data_url = "https://raw.githubusercontent.com/weaviate-tutorials/edu-datasets/main/movies_data_1990_2024.json"
resp = requests.get(data_url)
df = pd.DataFrame(resp.json())

# Loop through the dataset to generate vectors in batches
emb_dfs = list()
src_texts = list()
for i, row in enumerate(df.itertuples(index=False)):
    # Concatenate text to create a source string
    src_text = "Title" + row.title + "; Overview: " + row.overview
    # Add to the buffer
    src_texts.append(src_text)
    if (len(src_texts) == 50) or (i + 1 == len(df)):  # Get embeddings in batches of 50
        # Get a batch of embeddings
        output = vectorize(co, src_texts)
        index = list(range(i - len(src_texts) + 1, i + 1))
        emb_df = pd.DataFrame(output, index=index)
        # Add the batch of embeddings to a list
        emb_dfs.append(emb_df)
        # Reset the buffer
        src_texts = list()


emb_df = pd.concat(emb_dfs)  # Create a combined dataset

# Save the data as a CSV
os.makedirs("scratch", exist_ok=True)  # Create a folder if it doesn't exist
emb_df.to_csv(
    f"scratch/movies_data_1990_2024_embeddings.csv",
    index=False,
)

# Check Weaviate status
try:
    assert client.is_live()

    # Retrieve the server meta information
    metainfo = client.get_meta()
    print(json.dumps(metainfo, indent=2))

    # Create a movie collection.
    if client.collections.exists("MovieCustomVector"):
        print("Deleting existing MovieCustomVector collection.")
        client.collections.delete("MovieCustomVector")

    client.collections.create(
    name="MovieCustomVector",
    properties=[
        wc.Property(name="title", data_type=wc.DataType.TEXT),
        wc.Property(name="overview", data_type=wc.DataType.TEXT),
        wc.Property(name="vote_average", data_type=wc.DataType.NUMBER),
        wc.Property(name="genre_ids", data_type=wc.DataType.INT_ARRAY),
        wc.Property(name="release_date", data_type=wc.DataType.DATE),
        wc.Property(name="tmdb_id", data_type=wc.DataType.INT),
    ],
    # Define the vectorizer module
    vectorizer_config=wc.Configure.Vectorizer.none(),
    # Define the generative module
    generative_config=wc.Configure.Generative.cohere()
    )


    # Load the embeddings (embeddings from the previous step)
    # embs_path = "https://raw.githubusercontent.com/weaviate-tutorials/edu-datasets/main/movies_data_1990_2024_embeddings.csv"
    # Or load embeddings from a local file (if you generated them earlier)
    embs_path = "scratch/movies_data_1990_2024_embeddings.csv"

    emb_df = pd.read_csv(embs_path)

    # Get the collection
    movies = client.collections.get("MovieCustomVector")

    # Enter context manager
    with movies.batch.fixed_size(batch_size=200) as batch:
        # Loop through the data
        for i, movie in enumerate(df.itertuples(index=False)):
            # Convert data types
            # Convert a JSON date to `datetime` and add time zone information
            release_date = datetime.strptime(movie.release_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            # Convert a JSON array to a list of integers
            genre_ids = json.loads(movie.genre_ids)

            # Build the object payload
            movie_obj = {
                "title": movie.title,
                "overview": movie.overview,
                "vote_average": movie.vote_average,
                "genre_ids": genre_ids,
                "release_date": release_date,
                "tmdb_id": movie.id,
            }

            # Get the vector
            vector = emb_df.iloc[i].to_list()

            # Add object (including vector) to batch queue
            batch.add_object(
                properties=movie_obj,
                uuid=generate_uuid5(movie.id),
                vector=vector  # Add the custom vector
                # references=reference_obj  # You can add references here
            )
            # Batcher automatically sends batches
        
    # Check for failed objects
    if len(movies.batch.failed_objects) > 0:
        print(f"Failed to import {len(movies.batch.failed_objects)} objects")
    else:
        print(f'{movies.batch.failed_objects = }')

    # Perform query
    query_text = "dystopian future"
    query_vector = vectorize(co, [query_text])[0]  # Get the vector for the query text
    print(f"{query_text = }")
    response = movies.query.near_vector(
        limit=5, 
        return_metadata=wq.MetadataQuery(distance=True),
        near_vector=query_vector  # Use the custom vector for the query
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
        query="history", limit=5, return_metadata=wq.MetadataQuery(score=True),
        vector=query_vector  # Use the custom vector for the query
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
    response = movies.query.near_vector(
        near_vector=query_vector,  # Use the custom vector for the query
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

    response = movies.generate.near_vector(
        near_vector=query_vector,  # Use the custom vector for the query
        limit=5,
        single_prompt="Translate this into French: {title}"
    )

    # Inspect the response
    for o in response.objects:
        print(o.properties["title"])  # Print the title
        print(o.generated)  # Print the generated text (the title, in French)

    # Generative Search
    print("Grouped task prompt query: What do these movies have in common?")
    response = movies.generate.near_vector(
        near_vector=query_vector,  # Use the custom vector for the query
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