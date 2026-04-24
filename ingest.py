import os
import glob
import torch
import hashlib
import shutil

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from config import settings
from logger import logger


def get_file_hash(file_path):
    """计算文件的MD5值"""
    hasher = hashlib.md5()
    # 逐块读取文件 rb=二进制只读模式
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def ingest_documents(force_rebuild=False):
    """
    存入向量库
    :param force_rebuild: 是否强制重建（清空旧数据）
    """
    logger.info("开始处理文档...", force_rebuild={force_rebuild})

    docs_dir = settings.DOCS_DIR
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    # 获取已存在集合，查找比对之后决定重建还是新增
    try:
        collection = client.get_collection("rag_collection")
        if force_rebuild:
            client.delete_collection("rag_collection")
            collection = client.create_collection("rag_collection")
            logger.info("已清空旧数据库, 开始重新构建向量数据库...")
        else:
            logger.info("已找到旧数据库，开始追加向量数据库...")
    except:
        # 首次运行
        collection = client.create_collection("rag_collection")
        logger.info("已创建数据库, 开始构建向量数据库...")

    file_paths = (glob.glob(os.path.join(docs_dir, "*.txt")) +
                    glob.glob(os.path.join(docs_dir, "*.pdf")) +
                    glob.glob(os.path.join(docs_dir,"*.md")))

    if not file_paths:
        logger.warning("没有找到文件")
        return
    logger.info(f"找到 {len(file_paths)} 个文件")

    # 元数据对比
    existing_metadata = collection.get(include=["metadatas"])
    processed_files = set()
    if existing_metadata["metadatas"]:
        # 已处理过的文件名集合（过滤掉 None）
        processed_files = {
            meta.get("source_file") 
            for meta in existing_metadata['metadatas'] 
            if meta and meta.get("source_file")  # 加个判断
        }

    new_file = []
    updated_file = []
    unchanged_file = []

    for file_path in file_paths:
        abs_path = os.path.abspath(file_path)
        current_hash = get_file_hash(abs_path)  # 计算当前文件的MD5
        file_name = os.path.basename(file_path)

        if file_name not in processed_files:
            # 新文件
            new_file.append(file_path)
        else:
            # 已处理过的文件
            existing_docs = collection.get(
                where={"source_file": file_name},
                include=["metadatas"]
            )
            if existing_docs["metadatas"]:
                old_hash = existing_docs["metadatas"][0].get("file_hash")
                if old_hash != current_hash:
                    # 文件已存在，但内容已改变(MD5!=)
                    updated_file.append(file_path)
                    # 删除旧数据
                    collection.delete(where={"source_file": file_path})
                else:
                    # (MD5==,skip)
                    unchanged_file.append(file_path)
            else:
                new_file.append(file_path)

    logger.info(f"新增: {len(new_file)}, 更新: {len(updated_file)}, 未变化: {len(unchanged_file)}")


    files_to_process = new_file + updated_file
    if not files_to_process:
        logger.info("没有需要处理的文件")
        return


    documents = []
    file_hashes = {}

    for file_path in files_to_process:
        try:
            abs_path = os.path.abspath(file_path)
            file_hashes[os.path.basename(file_path)] = get_file_hash(abs_path)

            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")

            docs = loader.load ()

            # 思路：给每个文档片段添加元数据，方便后续追踪来源和去重
            for doc in docs:
                doc.metadata['source_file'] = os.path.basename(file_path)
                doc.metadata['file_hash'] = file_hashes[os.path.basename(file_path)]

            documents.extend(docs)
            logger.info(f"已处理文件: {file_path}")
        except Exception as e:
            logger.error(f"处理文件 {file_path} : {e}")

    if not documents:
        logger.warning("没有需要处理的文档")
        return

    # 切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE,
                                                   chunk_overlap=settings.CHUNK_OVERLAP,
                                                   length_function=len,
                                                   separators=["\n\n", "\n", "。", "；", "，", " ", ""])
    docs = text_splitter.split_documents(documents)
    logger.info(f"切分{len(docs)}个文档")

    # 提取文本和生成ID
    texts = [doc.page_content for doc in docs]
    ids = [f"{doc.metadata['source_file']}_{i}" for i, doc in enumerate(docs)]
    metadatas = [{
        'source_file': doc.metadata['source_file'],
        'file_hash': doc.metadata['file_hash']
    } for doc in docs]

    # 4.22修改，
    # 在写入前，先查询 Chroma 中是否已存在该 ID。如果存在，则跳过；
    # 如果不存在，再执行 add 操作。这能极大减少重复工作。
    # 批量计算 Embedding (GPU加速)
    logger.info("正在计算向量...")
    # 检测有没有 NVIDIA 显卡，有就用 GPU 加速计算向量，没有就用 CPU。
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
    logger.info(f"向量计算完成，维度: {embeddings.shape}")

    # 直接使用前面已获取的 collection，不要重新创建
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        ids=ids,
        metadatas=metadatas  # 记得加上元数据
    )

# 这个每一次都要清空，优化了(4.22
# # 清空并写入 Chroma
# if os.path.exists("./chroma_db"):
#     import shutil
#     shutil.rmtree("./chroma_db")
#     print("已清空旧数据库")

    # client = chromadb.PersistentClient(path="./chroma_db")
    # collection = client.create_collection("rag_collection")
    # collection.add(embeddings=embeddings.tolist(), documents=texts, ids=ids)

# 4.21原，Chroma.from_documents太慢了
# # 如果chroma数据库没跟新
# # Remove-Item -Recurse -Force .\chroma_db
# # python ingest.py
# for i, doc in enumerate(docs):
#     print(f"片段{i+1}长度: {len(doc.page_content)}, 内容预览: {doc.page_content[:50]}...")
#
# # 初始化Embedding模型（本地模型，免费）
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
# # 存入Charoma向量数据库  Chroma 在内部逐个计算 Embedding。串行，我去，这个怎么那么耗时
# vectorstore = Chroma.from_documents(documents = docs,
#                                     embedding = embeddings,
#                                     persist_directory="./chroma_db")
#
# vectorstore.persist()

    logger.info("向量数据库已保存")

if __name__ == "__main__":
    import sys
    # 支持命令行参数：python ingest.py --rebuild 强制重建
    force_rebuild = "--rebuild" in sys.argv
    ingest_documents(force_rebuild=force_rebuild)