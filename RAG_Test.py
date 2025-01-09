from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np


# 使用 Hugging Face 的 Transformers 作为嵌入模型
class TransformerEmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text):
        # 使用 Transformer 模型生成嵌入
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 获取最后一个隐藏层的池化输出作为嵌入向量
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        return embeddings


# 使用 FAISS 作为向量数据库
class FAISSVectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 距离索引
        self.documents = []  # 存储文档内容
        self.doc_id_map = {}  # 映射向量ID到文档ID

    def add_document(self, content):
        doc_id = len(self.documents)
        self.documents.append(content)
        return doc_id

    def add_vector(self, vector, doc_id):
        # 将向量添加到索引，同时记录向量到文档的映射
        self.index.add(np.array([vector]).astype("float32"))
        self.doc_id_map[self.index.ntotal - 1] = doc_id

    def search(self, query_vector, top_k=5):
        # 在 FAISS 中搜索最相似的向量
        query_vector = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        results = [
            {"content": self.documents[self.doc_id_map[idx]]} for idx in indices[0] if idx != -1
        ]
        return results


# 主查询引擎
class QueryEngine:
    def __init__(self, vector_store, embedding_model, language_model=None):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.language_model = language_model

    def query(self, query_text):
        # Step 1: 向量化查询
        query_vector = self.embedding_model.encode(query_text)

        # Step 2: 检索最相似文档
        results = self.vector_store.search(query_vector, top_k=1)

        # Step 3: 生成响应
        if self.language_model:
            # 高级生成方式
            prompt = f"用户问题: {query_text}\n相关文档:\n" + \
                     "\n".join([doc["content"] for doc in results])
            response = self.language_model.generate(prompt)
        else:
            # 简单拼接方式
            response = "\n".join([doc["content"] for doc in results])

        return response


# 测试代码
if __name__ == "__main__":
    # 初始化嵌入模型和向量存储
    embedding_model = TransformerEmbeddingModel()
    vector_store = FAISSVectorStore(dimension=384)  # MiniLM 模型输出的向量维度为 384
    query_engine = QueryEngine(vector_store, embedding_model)

    # 添加文档
    docs = [
        "Python是一种广泛使用的编程语言。",
        "机器学习是人工智能的一个分支。",
        "深度学习是机器学习的一种方法。",
        "Git是一种版本控制系统。",
        "LlamaIndex是一个基于Python的索引工具。"
    ]
    for doc in docs:
        doc_id = vector_store.add_document(doc)
        vector = embedding_model.encode(doc)
        vector_store.add_vector(vector, doc_id)

    # 执行查询
    query = "什么是git？"
    response = query_engine.query(query)
    print("查询结果:\n", response)
