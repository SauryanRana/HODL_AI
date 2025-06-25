import json
import chromadb
from sentence_transformers import SentenceTransformer
import time

def load_formatted_chunks(json_path):
    """Loads the formatted text chunks from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    return chunks_data

def main():
    # --- Configuration ---
    formatted_chunks_path = "hodltoken_formatted_chunks.json" # Input file from previous step
    chroma_db_path = "./hodl_chroma_db" # Directory for persistent ChromaDB storage
    collection_name = "hodl_content"
    embedding_model_name = 'all-mpnet-base-v2'

    print(f"Loading formatted chunks from: {formatted_chunks_path}")
    chunks_with_metadata = load_formatted_chunks(formatted_chunks_path)

    if not chunks_with_metadata:
        print("No chunks found. Exiting.")
        return

    print(f"Successfully loaded {len(chunks_with_metadata)} chunks.")

    print(f"Initializing embedding model: {embedding_model_name}")
    # Initialize the Sentence Transformer model
    # Make sure you have an internet connection the first time you run this
    # as it will download the model.
    model = SentenceTransformer(embedding_model_name)
    print("Embedding model initialized.")

    print(f"Initializing ChromaDB client (persistent storage at: {chroma_db_path})")
    # Initialize ChromaDB client.
    # Using a persistent client to save the database to disk.
    # CORRECT:
    client = chromadb.PersistentClient(path=chroma_db_path)

    # Get or create the collection
    print(f"Getting or creating ChromaDB collection: {collection_name}")
    # You can specify the distance function if needed, e.g., metadata={"hnsw:space": "cosine"}
    # By default, Chroma uses L2 squared distance, which is fine for many sentence embeddings.
    # Cosine similarity is also very common for sentence embeddings.
    # If using 'all-mpnet-base-v2', cosine similarity is generally preferred.
    # For SentenceTransformers, it's often good to use the distance metric the model was trained for.
    # all-mpnet-base-v2 produces normalized embeddings, so cosine similarity and dot product are equivalent
    # to L2 for ranking, but the actual distance values differ. Chroma's default is L2.
    # For normalized embeddings (like those from all-mpnet-base-v2), 'cosine' or 'ip' (inner product)
    # are good choices if you want to align with how similarity is often computed for these models.
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Specifying cosine distance
        )
        print(f"Collection '{collection_name}' ready.")
    except Exception as e:
        print(f"Error getting or creating collection: {e}")
        # If it's due to incompatible distance function with existing data, you might need to delete
        # the old collection or use the existing distance function.
        # For a first run, this should be fine.
        # To delete a collection if needed for a reset: client.delete_collection(name=collection_name)
        return

    # Prepare data for ChromaDB
    documents = []      # The text content of the chunks
    metadatas = []      # Metadata for each chunk (e.g., source URL)
    ids = []            # Unique IDs for each chunk
    embeddings_list = [] # List to store generated embeddings (optional if model.encode is fast enough for direct add)

    print(f"Preparing {len(chunks_with_metadata)} chunks for ChromaDB...")
    for i, chunk_data in enumerate(chunks_with_metadata):
        documents.append(chunk_data["text"])
        metadatas.append({
    "source_url": chunk_data.get("source_url", "N/A"),
    "original_chunk_id": chunk_data.get("chunk_id", "unknown_id")
})

        ids.append(chunk_data["chunk_id"]) # Using the chunk_id generated in format_data.py
        
        if (i + 1) % 100 == 0:
            print(f"Prepared {i+1}/{len(chunks_with_metadata)} chunks for metadata and IDs.")

    print("Generating embeddings for all documents...")
    start_time_embeddings = time.time()
    # Generate embeddings for all documents at once (more efficient)
    # The model.encode() method can take a list of strings.
    embeddings_generated = model.encode(documents, show_progress_bar=True)
    end_time_embeddings = time.time()
    print(f"Embeddings generated for {len(documents)} documents in {end_time_embeddings - start_time_embeddings:.2f} seconds.")


    # Add to ChromaDB in batches
    batch_size = 100 # Adjust batch size as needed based on your system's memory and performance
    print(f"Adding documents to ChromaDB in batches of {batch_size}...")
    num_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(documents))
        
        batch_documents = documents[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]
        batch_embeddings = embeddings_generated[start_idx:end_idx]

        print(f"Adding batch {i+1}/{num_batches} (chunks {start_idx+1}-{end_idx}) to collection...")
        try:
            collection.add(
                embeddings=batch_embeddings.tolist(), # Convert numpy array to list for ChromaDB
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"Batch {i+1} added successfully.")
        except Exception as e:
            print(f"Error adding batch {i+1} to ChromaDB: {e}")
            # You might want to add more sophisticated error handling or retry logic here
            
        time.sleep(0.1) # Small delay between batches if needed

    print("\n--- Knowledge Base Building Complete ---")
    try:
        collection_count = collection.count()
        print(f"Successfully added documents to the ChromaDB collection '{collection_name}'.")
        print(f"Total items in collection: {collection_count}")
        if collection_count == len(chunks_with_metadata):
            print("All chunks were added to the collection.")
        else:
            print(f"Warning: Expected {len(chunks_with_metadata)} chunks, but found {collection_count} in the collection.")
        
        # You can try a test query (optional)
        if collection_count > 0:
            print("\nPerforming a test query...")
            query_text = "What is HODL token?"
            query_embedding = model.encode(query_text).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=2 # Get top 2 results
            )
            print(f"Test query for '{query_text}':")
            if results and results.get('documents') and results['documents'][0]:
                for j, doc in enumerate(results['documents'][0]):
                    print(f"  Result {j+1}:")
                    print(f"    Text: {doc[:200]}...") # Print first 200 chars
                    print(f"    Metadata: {results['metadatas'][0][j]}")
                    if results.get('distances'):
                         print(f"    Distance: {results['distances'][0][j]}")
            else:
                print("  No results found for the test query or results format unexpected.")

    except Exception as e:
        print(f"Error verifying collection count or performing test query: {e}")
        
    print(f"\nChromaDB data is persisted in the '{chroma_db_path}' directory.")

if __name__ == '__main__':
    main()