import requests # pyright: ignore[reportMissingModuleSource]
import xml.etree.ElementTree as ET
import trafilatura # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient # pyright: ignore[reportMissingImports]
from qdrant_client.models import VectorParams, Distance, PointStruct # pyright: ignore[reportMissingImports]
import cohere # pyright: ignore[reportMissingImports]

# -------------------------------------
# CONFIG
# -------------------------------------
# Your Deployment Link:
SITEMAP_URL = "https://rag-doc.vercel.app/sitemap.xml"


COLLECTION_NAME = "humanoid_ai_book"

cohere_client = cohere.Client("api-key-here")
EMBED_MODEL = "embed-english-v3.0"

# Connect to Qdrant Cloud
qdrant_client = QdrantClient(
    url="https://345bd403-87a1-4060-8fce-8acc703c142e.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="api-key",
)

# -------------------------------------
# Step 1 — Extract URLs from sitemap
# -------------------------------------
def get_all_urls(sitemap_url):
    xml = requests.get(sitemap_url).text
    root = ET.fromstring(xml)

    urls = []
    for child in root:
        loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
        if loc_tag is not None:
            urls.append(loc_tag.text)

    print("\nFOUND URLS:")
    for u in urls:
        print(" -", u)

    return urls


# -------------------------------------
# Step 2 — Download page + extract text
# -------------------------------------
def extract_text_from_url(url):
    html = requests.get(url).text
    text = trafilatura.extract(html)

    if not text:
        print("[WARNING] No text extracted from:", url)

    return text


# -------------------------------------
# Step 3 — Chunk the text
# -------------------------------------
def chunk_text(text, max_chars=1200):
    chunks = []
    while len(text) > max_chars:
        split_pos = text[:max_chars].rfind(". ")
        if split_pos == -1:
            split_pos = max_chars
        chunks.append(text[:split_pos])
        text = text[split_pos:]
    chunks.append(text)
    return chunks


# -------------------------------------
# Step 4 — Create embedding
# -------------------------------------
def embed(text):
    response = cohere_client.embed(
        model=EMBED_MODEL,
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding


# -------------------------------------
# Step 5 — Store in Qdrant
# -------------------------------------
def create_collection():
    print("\nCreating Qdrant collection...")
    qdrant.recreate_collection( # pyright: ignore[reportUndefinedVariable]
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
        size=1024,        # Cohere embed-english-v3.0 dimension
        distance=Distance.COSINE
        )
    )

def save_chunk_to_qdrant(chunk, chunk_id, url):
    vector = embed(chunk)

    qdrant.upsert( # pyright: ignore[reportUndefinedVariable]
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "url": url,
                    "text": chunk,
                    "chunk_id": chunk_id
                }
            )
        ]
    )


# -------------------------------------
# MAIN INGESTION PIPELINE
# -------------------------------------
def ingest_book():
    urls = get_all_urls(SITEMAP_URL)

    create_collection()

    global_id = 1

    for url in urls:
        print("\nProcessing:", url)
        text = extract_text_from_url(url)

        if not text:
            continue

        chunks = chunk_text(text)

        for ch in chunks:
            save_chunk_to_qdrant(ch, global_id, url)
            print(f"Saved chunk {global_id}")
            global_id += 1

    print("\n✔️ Ingestion completed!")
    print("Total chunks stored:", global_id - 1)


if __name__ == "__main__":
    ingest_book()
