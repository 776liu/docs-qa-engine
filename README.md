# RAG API demo

基于 FastAPI + ChromaDB + DeepSeek 的本地知识库问答系统。

## 核心功能
- 📄 支持 PDF/TXT 文档上传与自动入库
- 🔍 语义检索 + LLM 生成答案
- ⚡ 增量更新（MD5 去重）

## ✅ 已完成
    RAG 核心链路：文档加载 → 切分 → 向量化 → 检索 → 生成
    多格式支持：PDF + TXT
    增量更新：MD5 去重，只处理变化文件
    API 接口：
    POST /ask（问答）
    POST /upload（上传并自动入库）
    GET /（健康检查）
    性能优化：延迟加载、离线模式、批量计算

## 快速启动

### 1. 安装依赖
    - bash pip install -r requirements.txt

    复制 `.env.example` 为 `.env`，填入你的 DeepSeek Key。

### 2. 启动服务
    bash 

    python -m uvicorn main:app --reload

### 3. 访问文档
    http://127.0.0.1:8000/docs

## 项目结构
    RAG_demo\
    ├── main.py     : API 路由
    ├── rag_engine.py   : RAG 核心逻辑
    ├── ingest.py   : 文档处理与入库
    