import os
import json
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
load_dotenv() # 加载.env文件中的环境变量
import numpy as np
from datasets import Dataset  # noqa: E402
from ragas.metrics import Faithfulness, AnswerRelevancy  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from ragas import evaluate
import sqlparse
import sqlite3
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# 配置
os.environ["OPENAI_API_KEY"] = (
    ""  # 替换为您的 API key
)
DB_PATH = "/Volumes/Extend/self_workspaces/python/rag-in-action/09-Evaluation/data/sakila.db"  # SQLite 数据库路径
JSON_PATH = "/Volumes/Extend/self_workspaces/python/rag-in-action/09-Evaluation/data/q2sql_pairs.json"  # JSON 文件路径
OPENAI_BASE_URL = "https://api.gptsapi.net/v1"

SAKILA_SCHEMA = """
Tables:
- actor: actor_id (PK), first_name, last_name, last_update
- film: film_id (PK), title, description, release_year, language_id, rental_duration, rental_rate, length, replacement_cost, rating, last_update
- customer: customer_id (PK), store_id, first_name, last_name, email, address_id, active, create_date, last_update
- rental: rental_id (PK), rental_date, inventory_id, customer_id (FK), return_date, staff_id, last_update
- inventory: inventory_id (PK), film_id (FK), store_id, last_update
- payment: payment_id (PK), customer_id (FK), staff_id, rental_id, amount, payment_date, last_update
- category: category_id (PK), name, last_update
- film_actor: actor_id (FK), film_id (FK)
- store: store_id (PK), manager_staff_id, address_id, last_update
- staff: staff_id (PK), first_name, last_name, address_id, email, store_id, active, username, last_update
Relationships: film -> inventory -> rental -> customer; film -> film_actor -> actor; customer -> payment; film -> category; store -> staff
"""


def main0():
    llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-3.5-turbo", base_url=OPENAI_BASE_URL))


    with open(JSON_PATH, "r") as f:
        q2sql_pairs = json.load(f)

    data = {
        "question": [],
        "answer": [],
    }
    
    for items in q2sql_pairs:
        template="""Based on the following database schema: {schema}
        
            Generate SQL for: {question}
        
            SQL:"""
        question = template.format(schema=SAKILA_SCHEMA, question=items["question"])
        data["question"].append(question)
        data["answer"].append(items["sql"])

    data["retrieved_contexts"] = [[] for _ in range(len(data["question"]))]

# 将字典转换为Hugging Face的Dataset对象，方便Ragas处理
    dataset = Dataset.from_dict(data)

    print("\n1. AnswerRelevancy（答案相关性）")
    print("- 评估生成的答案与问题的相关程度")
    print("- 使用embedding模型计算语义相似度")

# 2. OpenAI的 text-embedding-ada-002 模型
    openai_embedding = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-ada-002", base_url=OPENAI_BASE_URL))

# 创建答案相关性评估指标
# 分别为两种embedding模型创建AnswerRelevancy评估指标
    openai_relevancy = [AnswerRelevancy(llm=llm, embeddings=openai_embedding)]

    print("\n使用OpenAI Embedding模型评估:")
# 使用OpenAI embedding模型进行评估
    openai_result = evaluate(dataset, openai_relevancy)
    scores = openai_result['answer_relevancy']
    openai_mean = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
    print(f"相关性评分: {openai_mean:.4f}")

def generate_predicted_sql(question: str) -> str:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        base_url=OPENAI_BASE_URL
    )
    prompt = PromptTemplate(
        input_variables=["question", "schema"],
        template="""Based on the following database schema: {schema}
        
        Generate SQL for: {question}
        
        SQL:"""
    )
    chain = prompt | llm
    response = chain.invoke({"question": question, "schema": SAKILA_SCHEMA})
    text_sql =  response.content.replace("\n", " ").strip()
    #print(f"Generated SQL: {text_sql}")
    return text_sql


def normalize_sql(sql):
    """规范化SQL：去除多余空格，小写"""
    return sqlparse.format(sql, reindent=False, keyword_case='lower').strip()

def execute_sql(db_path, sql):
    """执行SQL并返回结果作为DataFrame"""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except sqlite3.Error as e:
        print(f"SQL执行错误: {sql} - {e}")
        return None
    finally:
        conn.close()

def compute_metrics(gold_sql, generated_sql, db_path):
    metrics = {}
    norm_gold = normalize_sql(gold_sql)
    norm_gen = normalize_sql(generated_sql)
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_scores = scorer.score(norm_gold, norm_gen)
    metrics['ROUGE-L'] = rouge_scores['rougeL'].fmeasure
    
    # BLEU
    metrics['BLEU'] = sentence_bleu([norm_gold.split()], norm_gen.split())

    return metrics

def main1(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
      
    #data = data[:3]  
    
    total_metrics = {'ROUGE-L': 0, 'BLEU': 0}
    num_samples = len(data)
    
    for sample in data:
        generated_predicted_sql = generate_predicted_sql(sample['question'])
        metrics = compute_metrics(sample['sql'], generated_predicted_sql, DB_PATH)
        for key in total_metrics:
            total_metrics[key] += metrics.get(key, 0)
    
    # 计算平均
    avg_metrics = {k: v / num_samples for k, v in total_metrics.items()}
    print("平均指标:", avg_metrics)
    
    # 可选: 输出到CSV
    pd.DataFrame([avg_metrics]).to_csv('metrics_report.csv', index=False)
    
    
if __name__ == "__main__":
    main0()
    main1(JSON_PATH)
    # main1("path_to_your_generated_sql_dataset.json")  # 替换为实际路径
