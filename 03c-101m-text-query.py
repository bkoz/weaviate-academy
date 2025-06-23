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
    response = movies.query.near_text(
        query="red",
        limit=5,
        return_metadata=wq.MetadataQuery(distance=True),
        return_properties=["title", "release_date", "tmdb_id", "poster"]  # To include the poster property in the response (`blob` properties are not returned by default)
    )

    # Inspect the response
    for o in response.objects:
        print(
            o.properties["title"], o.properties["release_date"].year, o.properties["tmdb_id"]
        )  # Print the title and release year (note the release date is a datetime object)
        print(
            f"Distance to query: {o.metadata.distance:.3f}\n"
        )  # Print the distance of the object from the query

finally:
    client.close()