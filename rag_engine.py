import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

class RAGEngine:
    def __init__(self):
        self.embed_model = None
        self.collection = None
        self.llm = None
        self._initialized = False

    def initialize(self):
        """延迟初始化: 不在启动时加载模型，而是等第一次请求时才加载。这样服务秒开。"""
        if self._initialized:
            return
        print("正在加载 Embedding 模型...")

        start = time.time()
        self.embed_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            model_kwargs={"local_files_only": True}
        )
        print(f"模型加载完成，耗时: {time.time() - start:.2f}秒")

        # 连接向量数据库
        client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = client.get_collection("rag_collection")
        print("向量数据库连接成功")

        # 创建 LLM
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0
        )

        self._initialized = True
        print(f"RAG 引擎初始化完成")

    def ask(self, question: str, top_k: int = 3) ->dict:
        """ 回答接口"""
        self.initialize()

        # 计算问题向量
        query_embedding = self.embed_model.encode(question).tolist()

        # 检索相似文档
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        docs_content = results['documents'][0]
        print(f"检索到 {len(docs_content)} 个文档片段:")
        for i, content in enumerate(docs_content):
            print(f"片段{i + 1}: {content[:100]}...")

        # 组装RAG链
        context = "\n\n".join(results['documents'][0])
        prompt = f"资料：\n{context}\n\n问题：{question}\n回答："
        return {"answer": self.llm.invoke(prompt).content}

