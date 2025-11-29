import redis
import json
import duckdb
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from google import genai
from pydantic import BaseModel
import os
import json
import gradio as gr

from dotenv import load_dotenv

load_dotenv()

google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-embedding-001"
)

client = chromadb.PersistentClient('db/workshop_1')

collection = client.get_collection("gemini_demo", embedding_function=google_ef)

db = duckdb.connect("db/workshop_db")
table_name = "alcohol_price"

if not db.sql(f"SELECT * FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone():
    db.sql(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('sample csv.csv')")

df = db.sql(f"DESCRIBE {table_name}")

r = redis.Redis(host='localhost', port=6379)

llm = genai.Client(
    api_key = os.getenv('GEMINI_API_KEY'),
)

def get_history(user_id):
    messages = r.lrange(f"chat:{user_id}",0,-1)
    return [json.loads(m.decode()) for m in messages]

def add_history(user_id, role, messages):
    r.rpush(f"chat:{user_id}", json.dumps({"role": role, "content": messages}))

def rewrite_query(query, conversations: list[dict]):
    history_context = "\n".join([f"{conv['role']}: {conv['content']}" for conv in conversations])

    prompt = f"""
    The conversation context is
    {history_context}
    Query:
    {query}
    """

    response = llm.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt,
        config = {
            "system_instruction": "You are helpful assistant that can rewrite queries to be more accurate. Always put the name of the person or produt if the had mentioned.",
            "temperature": "0"
        }
    )

    return response.text

def get_rag_answer(results, query, model, model_name="gemini-2.5-flash", temperature=0):
    context = "\n".join(results)

    prompt = f"""
    Use the following context to answer the question.
    If you cannot answer the question based on the context, please answer "I can't answer it based on the following context".
    Context:
    {context}

    Query:
    {query}
    """

    response = llm.models.generate_content(
        model = f"{model_name}",
        contents = prompt,
        config = {
            "system_instruction": "You are helpful assistant that can rewrite queries to be more accurate. Always put the name of the person or produt if the had mentioned.",
            "temperature": f"{temperature}"
        }
    )

    return response.text

def retrieve_query(query):
    results = collection.query(
        query_texts = query,
        n_results = 3
    )

    return get_rag_answer(results['documents'][0], query, client)

def text_2_db(query):
    table_name = 'alcohol_price'
    df = db.sql(f"DESCRIBE {table_name}")

    prompt = f"""
        You are a helpful assistant that can generate SQL queries that answer the question about the data.
        Please do select it from table {table_name}
        Also do notes that the column of the {table_name} were:
        {df}

        Here are the question for you to answer: {query}
    """

    response = llm.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config = {
            "system_instruction": "You are a helpful assistant that can generate SQL queries. Only return the SQL and nothing more",
            "temperature": "0"
        }
    )

    sql_query = response.text
    sql_query = sql_query.replace("```sql","").replace("```","")

    res = db.sql(sql_query)
    return res

def classify_intent(query):
    prompt = f"""
    Decide if the query needs a vector search or text 2 sql search.print
    If the user is asking for specific data which is usually like a numerical operation like greater than, less than, etc. then it is a text 2 sql search. Similary if they are asking for details mentioning a specific field name like find me details of employee with id 123 then it is a text 2 sql search.
    If the user query is a general, hi, hello, etc. then it is a General intent.
    For everything else, it is a vector search.

    Here are the list of tables in the database:
    {db.sql("SELECT table_name FROM information_schema.tables").fetchall()}

    here is the list of pdf documents we have ingested in vector db:
    "sample pdf.pdf" regarding the danger of alcohol
    "sample pdf 1.pdf" regarding pros and cons of alcohol
    "sample pdf 2.pdf" regarding health concious guide of alcohol

    The query is: {query}
    Only return the Intent as a string. It will be either "vector" or "text_2_db" or "general".
    """

    response = llm.models.generate_content(
        model="gemini-2.5-flash",
        contents = prompt,
        config = {
            "temperature": "0"
        }
    )

    return response.text

def run_agentic_rag(query, user_id):
    add_history(user_id, "user", query)
    rewrite = rewrite_query(query, get_history(user_id))
    print(rewrite)
    intent = classify_intent(rewrite)
    print(intent)
    if intent == "general":
        return "No intention, can't answer question"
    elif intent == "text_2_db":
        return text_2_db(rewrite)
    elif intent == "vector":
        return retrieve_query(rewrite)

def chat_interface(message, history):
    user_id = "user_id_1"
    try:
        response = run_agentic_rag(message, user_id)
        if not isinstance(response, str):
            response = str(response) if response else "No response"
        return response
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    gr.ChatInterface(fn=chat_interface, title="Chat").launch()