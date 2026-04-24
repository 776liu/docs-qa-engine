import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from typing import List, Dict, Optional, Generator
import hashlib
from config import settings
from logger import logger

class RAGEngine:
    def __init__(self):
        self.embed_model = None
        self.collection = None
        self.llm = None
        self._initialized = False

        # 保存对话历史 {session_id: [messages]}
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

        # 答案缓存 {question_hash: answer}
        self.answer_cache: Dict[str, str] = {}
        self.max_cache_size = settings.MAX_CACHE_SIZE

    def initialize(self):
        """延迟初始化: 不在启动时加载模型，而是等第一次请求时才加载。这样服务秒开。"""
        if self._initialized:
            return
        logger.info("正在加载 Embedding 模型...")
        start = time.time()

        try:
            self.embed_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device="cpu",
                model_kwargs={"local_files_only": True}
            )
            logger.info(f"模型加载完成，耗时: {time.time() - start:.2f}秒")
        except Exception as e:
            logger.error(f"加载 Embedding 模型失败: {e}")
            raise

        try:
            # 连接向量数据库
            client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = client.get_collection("rag_collection")
            logger.info("向量数据库连接成功")
        except Exception as e:
            logger.error(f"Embedding 模型加载失败: {str(e)}", exc_info=True)
            raise


        try:
            # 创建 LLM
            self.llm = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                openai_api_base="https://api.deepseek.com/v1",
                temperature=0
            )
            logger.info("LLM 初始化成功")
        except Exception as e:
            logger.error(f"LLM 初始化失败: {str(e)}", exc_info=True)
            raise

        self._initialized = True
        print(f"RAG 引擎初始化完成")


    def _get_cache_key(self, question: str, session_id: Optional[str] = None) -> str:
        """
        生成缓存键值

        :param question:
        :param session_id:
        :return: MD5 str
        """

        cache_content = question

        # 有session_id, 则加入会话历史
        if session_id and session_id in self.sessions:
            history_str = "\n".join(
                [f"{msg["role"]}: {msg["content"]}" for msg in self.sessions[session_id][-5:]]
                )
            cache_content = f"{history_str}\n{question}"
        # 生成固定长度的key，MD5 hexdigest()把提问和历史对话组合为str
        return hashlib.md5(cache_content.encode("utf-8")).hexdigest()

    def _check_cache(self,cache_key: str)-> Optional[str]:
        """
        检索前先查缓存，命中直接返回

        :param cache_key:
        :return: 返回命中的字符串
        """
        return self.answer_cache.get(cache_key)

    def _save_to_cache(self, cache_key: str, answer: str):
        """
        保存答案
        用问题文本作为 key，答案作为 value
        先检查缓存的大小，删除部分历史条目后载入缓存

        :param cache_key:
        :param answer:
        :return:
        """
        if len(self.answer_cache) >= self.max_cache_size:
            keys_to_remove = list(self.answer_cache.keys())[:self.max_cache_size // 2]
            for key in keys_to_remove:
                del self.answer_cache[key]
            print(f"缓存已满，清理了{len(keys_to_remove)}条记录")

        self.answer_cache[cache_key] = answer

    def  _build_prompt_with_history(self, question: str, context: str,
                                   session_id: Optional[str] = None) -> str:
        """
        构建包含对话历史的prompt

        :param question:
        :param context:
        :param session_id:
        :return: prompt字符串
        """
        prompt_parts = []

        prompt_parts.append("你是一个智能助手，基于提供的资料回答问题。如果资料中没有相关信息，请如实告知。")
        prompt_parts.append(f"\n\n参考: \n {context}")

        if session_id and session_id in self.sessions:
            history = self.sessions[session_id][-5:]
            if history:
                prompt_parts.append("\n\n对话历史:")
                for msg in history:
                    role_cn = "user" if msg["role"] == "user" else "assistant"
                    prompt_parts.append(f"{role_cn}: {msg['content']}")

        prompt_parts.append(f"\n\n问题: {question}\n回答:")
        return "\n".join(prompt_parts)

    def ask(self, question: str, top_k: int = None,session_id: Optional[str] = None) ->dict:
        """ 回答接口(非流式)"""
        if top_k is None:
            top_k = settings.TOP_K

        self.initialize()
        logger.info(f"收到问答请求: session_id={session_id}, question_length={len(question)}")

        # 检查缓存
        cache_key = self._get_cache_key(question, session_id)
        cached_answer = self._check_cache(cache_key)
        if cached_answer:
            # 命中缓存,直接输出相同回答
            return {"answer": cached_answer, "session_id": session_id, "from_cache": True}

        try:
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
            prompt = self._build_prompt_with_history(question, context, session_id)
            # llm
            answer = self.llm.invoke(prompt).content
            self._save_to_cache(cache_key, answer)

            if session_id:
                if session_id not in self.sessions:
                    self.sessions[session_id] = []
                self.sessions[session_id].append({"role": "user", "content": question})
                self.sessions[session_id].append({"role": "assistant", "content": answer})

                if len(self.sessions[session_id]) > 20:
                    self.sessions[session_id] = self.sessions[session_id][-20:]

            return {"answer": answer, "session_id": session_id, "from_cache": False}

        except Exception as e:
            logger.error(f"回答处理失败:{str(e)}", exc_info=True)
            raise

    # Generator[str, None, None] 产出str，发送None，返回None
    def ask_stream(self,question: str, top_k: int = None,
                   session_id: Optional[str] = None) -> Generator[str, None, None]:
        """
        流式回答接口

        :param question:
        :param top_k:
        :param session_id:
        :return: 生成器，逐个yield token字符串

        """
        if top_k is None:
            top_k = settings.TOP_K

        self.initialize()

        cache_key = self._get_cache_key(question, session_id)
        cached_answer = self._check_cache(cache_key)
        if cached_answer:
            # 命中缓存,直接输出相同回答
            for cache in cached_answer:
                yield cache
            return

        try:
            query_embedding = self.embed_model.encode(question).tolist()

            results = self.collection.query(
                query_embeddings = [query_embedding],
                n_results = top_k
            )

            docs_content = results['documents'][0]
            print(f"检索到 {len(docs_content)} 个文档片段:")

            context = "\n\n".join(results['documents'][0])
            prompt = self._build_prompt_with_history(question, context, session_id)

            full_answer = []
            for chunk in self.llm.stream(prompt):
                token = chunk.content
                full_answer.append(chunk)
                yield token

            answer = "".join(full_answer)
            self._save_to_cache(cache_key, answer)

            if session_id:
                if session_id not in self.sessions:
                    self.sessions[session_id] = []
                self.sessions[session_id].append({"role": "user", "content": question})
                self.sessions[session_id].append({"role": "assistant", "content": answer})

                if len(self.sessions[session_id]) > 20:
                    self.sessions[session_id] = self.sessions[session_id][-20:]
        except Exception as e:
            logger.error(f"流式回答处理失败:{str(e)}", exc_info=True)
            raise

    def clear_session(self,session_id: str) -> None:
        """
        清空会话历史
        :param session_id:
        :return: TURE,FALSE
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def get_session_history(self,session_id: str) -> List[Dict[str, str]]:
        """
        获取历史
        :param session_id:
        :return: [{"role": "user/assistant", "content": "..."}]
        """
        return self.sessions.get(session_id, [])