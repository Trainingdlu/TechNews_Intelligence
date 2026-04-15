import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# 引入 Ragas 相关模块 (适配 Ragas v0.3.x)
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import sys
import argparse

# 确保能正确引入项目的其余模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.db import init_db_pool, db_cursor

def load_sandbox_documents(limit: int = 5000):
    """
    直接连接到 PostgreSQL 数据库，抽取最近的 N 条新闻用于生成测试问题。
    """
    print(f"正在通过数据库连接池提取最近的 {limit} 条新闻文段")
    init_db_pool()
    
    docs = []
    try:
        from contextlib import closing
        with db_cursor() as (conn, cur):
            query = """
                SELECT title, summary, url, created_at 
                FROM tech_news 
                WHERE summary IS NOT NULL AND length(summary) > 20
                ORDER BY created_at DESC 
                LIMIT %s
            """
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            
            for row in rows:
                # row: (title, summary, url, created_at)
                title, summary, url, created_at = row
                
                # 拼接标题与摘要作为 RAGAS 的有效文档池
                content = f"【标题】{title}\n【摘要/正文】{summary}"
                
                # 保留元数据，便于追踪来源
                meta = {
                    "source": url,
                    "title": title,
                    "date": str(created_at)
                }
                docs.append(Document(page_content=content, metadata=meta))
                
        print(f"成功从 PostgreSQL 的 'tech_news' 表拉取了 {len(docs)} 篇新鲜科技新闻！")
    except Exception as e:
        print(f"数据库连接/查询异常: {e}")
        sample_texts = [
            "2026年4月微软发布了Phi-4系列小语言模型，该系列最高具备140亿参数，以不到GPT-4十分之一的算力成本即可在其特定的代码评估集上达到持平的性能。",
            "OpenAI今日宣布GPT-5已经完成早期红蓝对抗测试，据称其引入了原生的多模态骨干网络，能够同时理解并深度模拟音频和3D空间信息。"
        ]
        docs = [Document(page_content=text, metadata={"source": f"db_fallback_{i}", "topic": "AI fallback"}) for i, text in enumerate(sample_texts)]
    
    return docs

def generate_ragas_testset():
    """使用 Ragas 生成测试集，输出 CSV"""
    # 建立生成及审核模型 (统一采用 vertexai)
    generator_llm = ChatVertexAI(
        model_name="gemini-3-flash",
        temperature=0.7,
        max_retries=3
    )

    critic_llm = ChatVertexAI(
        model_name="gemini-3.1-pro-preview",
        temperature=0.0,
        max_retries=3
    )

    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004"
    )

    wrapped_generator = LangchainLLMWrapper(generator_llm)
    wrapped_critic = LangchainLLMWrapper(critic_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Ragas v0.3.x：直接使用 from_langchain，审核/提取模型通过 transforms_llm 传递
    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings
    )

    # 抽取最近的 5000 条真实新闻数据放入 LLM 的问题生发池
    docs = load_sandbox_documents(limit=5000)

    # 设定生成的数据分布
    # 说明：单事实提取(simple)，逻辑推理(reasoning)，多文档综合(multi_context)

    test_size = 50 # Baseline size
    print(f"🚀 generating {test_size} questions with gemini-3/3.1")
    
    os.environ["RAGAS_MAX_RETRIES"] = "3"
    
    # 彻底关闭 LangSmith Tracing 防止 Ragas 和 LLM 强行把几十万个 Sub-span 都塞到免费版限制上导致 429 报错打断测试集生成
    # 或者如果被设置为了 true，我们也用 python os.environ 屏蔽
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Ragas 0.3.x 接口
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=test_size,
        transforms_llm=critic_llm,
        with_debugging_logs=True
    )

    df = testset.to_pandas()
    output_path = "eval/datasets/generated_rag_testset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"测试集已导出至：{output_path}")
    print(f"数据结构预览：\n{df[['question', 'ground_truth']].head(3)}")

if __name__ == "__main__":
    # 加载系统环境变量（需确保 GCP 鉴权已设置）
    load_dotenv("agent/.env")
    generate_ragas_testset()
