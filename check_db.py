from chromadb import PersistentClient

client = PersistentClient(path="./chroma_db")
collection = client.get_collection("rag_collection")

docs = collection.get()
print(f"总文档数: {len(docs['ids'])}")
for i, (doc_id, content) in enumerate(zip(docs['ids'], docs['documents'])):
    print(f"\n文档{i+1} (ID: {doc_id}):")
    print(content[:200])
