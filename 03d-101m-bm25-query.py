import weaviate
import weaviate.classes.query as wq
import os


# 
# Connect to Weaviate
#
headers = {
    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
}

client = weaviate.connect_to_local()

# Check Weaviate status
try:
    assert client.is_live()

    # Get the collection
    movies = client.collections.get("MovieMM")

    # Perform query
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


finally:
    client.close()