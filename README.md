# RAG API demo

基于 FastAPI + ChromaDB + DeepSeek 的本地知识库问答系统。

## 核心功能
- 📄 支持 PDF/TXT/Markdown 文档上传与自动入库
- 🔍 语义检索 + LLM 生成答案
- 💬 多轮对话历史管理
- ⚡ 增量更新（MD5 去重）
- 🔐 API Token 鉴权
- 📝 完整日志记录
- 🏗️ 分层架构设计

## 技术栈
- **Web 框架**: FastAPI
- **向量数据库**: ChromaDB
- **Embedding 模型**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: DeepSeek API
- **文档处理**: LangChain

## API 接口：

### 1. 健康检查
    bash 
    GET /

### 2. 非流式问答
    bash 
    POST /ask 
    Headers: Authorization: Bearer <your-token> 
    Body: { 
        "question": "(输入文档相关的问题)?",
        "session_id": "optional-session-id" 
    }

### 3. 流式问答
    bash 
    POST /ask/stream 
    Headers: Authorization: Bearer <your-token> 
    Body: { 
    "question": "(输入文档相关的问题)", 
    "session_id": "optional-session-id" 
    }

### 4. 获取会话历史
    bash 
    GET /session/{session_id}/history 
    Headers: Authorization: Bearer <your-token>

### 5. 上传文档
    bash 
    POST /upload 
    Headers: Authorization: Bearer <your-token> 
    Form-data: file (支持 .pdf, .txt, .md)


## 快速启动

### 1. 环境准备
    创建虚拟环境
    python -m venv .venv

    激活虚拟环境
    Windows:
    .venv\Scripts\activate
    Linux/Mac:
    source .venv/bin/activate

    安装依赖
    pip install fastapi uvicorn chromadb sentence-transformers langchain langchain-openai langchain-community pypdf python-dotenv pydantic-settings python-multipart

### 2. 配置环境变量
    复制 `.env` 文件并填写配置：
    bash DEEPSEEK_API_KEY=sk-your-deepseek-key 
    API_TOKEN=your-secret-token-here 
    LOG_LEVEL=INFO 
    CHROMA_DB_PATH=./chroma_db 
    DOCS_DIR=./docs

### 3. 启动服务
    bash 
    python -m uvicorn main:app --reload

### 4. 访问文档
    浏览器打开：http://127.0.0.1:8000/docs
    
    在 Swagger UI 中点击右上角 **Authorize**，输入 Token 即可测试接口。
    
## 项目结构
    RAG_demo/
    ├── routes/
    │   ├── __init__.py
    │   ├── chat.py             # 问答相关路由
    │   └── upload.py           # 文件上传路由
    ├── services/
    │   ├── __init__.py
    │   └── rag_service.py      # RAG 业务逻辑
    ├── dependencies/
    │   ├── __init__.py
    │   └── auth.py             # 鉴权中间件
    ├── config.py               # 配置管理
    ├── logger.py               # 日志模块
    ├── rag_engine.py           # RAG 核心引擎
    ├── ingest.py               # 文档入库处理
    └── main.py

## 更新日志

-   ### 4.23新增
    - Markdown 支持：新增可上传 .md 文档
    - 流式输出，逐字 streaming
    - 对话历史：添加 session 管理，每次都是独立问答
    - 缓存：相同问题会重复检索和调用 LLM
    
-   ### 4.24新增
    - 添加logger日志记录    
    - 添加API的Token鉴权认证
    - 添加config管理硬编码路径
    - 完成了逻辑分离







