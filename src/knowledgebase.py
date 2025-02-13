import chromadb
import os

def initialize_chroma_knowledge_base(db_path='knowledge_base'):

    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print("Creating new Chroma knowledge base...")
    else:
        print("Chroma knowledge base already exists, loading...")

    client = chromadb.PersistentClient(path=db_path)

    return client

def get_openai_embedding(text, openai_client):

    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def list_collection_data(client):
    collection_names = client.list_collections()

    for collection_name in collection_names:
        print(f"Collection Name: {collection_name}")

        collection = client.get_collection(name=collection_name)

        data = collection.get(include=['documents', 'metadatas'])

        for doc, meta, doc_id in zip(data['documents'], data['metadatas'], data['ids']):
            print(f"Document ID: {doc_id}")
            print(f"Metadata: {meta}")
            print(f"Content: {doc}\n")

def add_to_chroma(client, company_name, keyword, document, openai_client):

    collection_name = company_name.lower().replace(" ", "_")  # e.g., "BK Connect" -> "bk_connect"
    collection = client.get_or_create_collection(name=collection_name)

    doc_id = keyword.lower().replace(" ", "_")  # e.g., "Human Vibration" -> "human_vibration"

    existing_docs = collection.get(ids=[doc_id])
    if existing_docs["ids"]:
        print(f"Document for '{keyword}' already exists in '{company_name}'. Skipping.")
        return

    embedding = get_openai_embedding(document, openai_client)

    collection.add(
        documents=[document],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas={"keyword": keyword, "company": company_name}
    )

    print(f"Added document for '{keyword}' under company '{company_name}'.")

def retrieve_context_from_chroma(client, company_name, keyword, n_results=1, similarity_threshold=0.5, openai_client=None):

    collection_name = company_name.lower().replace(" ", "_")
    collection = client.get_or_create_collection(name=collection_name)

    query_embedding = get_openai_embedding(keyword, openai_client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['distances', 'documents', 'metadatas']
    )

    relevant_docs = [
        doc for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        if distance <= similarity_threshold and keyword.lower() in meta.get('keyword', '').lower()
    ]

    return relevant_docs

def clear_chroma_collection(client, collection_name, full_clear = False):

    try:
        collection = client.get_collection(name=collection_name)

        data = collection.get(include=["documents"])  # Corrected include

        document_ids = data.get("ids", [])

        if document_ids:
            collection.delete(ids=document_ids)
            print(f"Cleared {len(document_ids)} documents from collection '{collection_name}'.")
        else:
            print(f"No documents found in collection '{collection_name}' to clear.")

        if full_clear:
            clear_chroma_collection(client, collection_name)

    except Exception as e:
        print(f"Error clearing collection '{collection_name}': {e}")

def delete_chroma_collection(client, collection_name):
    try:
        client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting collection '{collection_name}': {e}")

def add_meeting_history(client, company_name, meeting_summary, timestamp, openai_client):

    collection_name = f"{company_name.lower().replace(' ', '_')}_meetings"
    collection = client.get_or_create_collection(name=collection_name)
    doc_id = f"meeting_{timestamp}"

    embedding = get_openai_embedding(meeting_summary, openai_client)
    collection.add(
        documents=[meeting_summary],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas={"timestamp": timestamp, "company": company_name}
    )
    print(f"Added meeting history for '{company_name}' with ID '{doc_id}'.")

def retrieve_latest_meeting_summary(client, company_name):

    collection_name = f"{company_name.lower().replace(' ', '_')}_meetings"
    collection = client.get_or_create_collection(name=collection_name)
    data = collection.get(include=["documents", "metadatas"])
    if data and data.get("documents"):
        # Sort the documents by timestamp (assumes timestamp is stored in metadata)
        sorted_docs = sorted(
            zip(data["documents"], data["metadatas"]),
            key=lambda x: x[1].get("timestamp", 0),
            reverse=True
        )
        latest_summary = sorted_docs[0][0]  # Return the latest meeting summary
        print(f"Retrieved latest meeting summary for '{company_name}'.")
        return latest_summary
    return None