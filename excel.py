from sqlalchemy import create_engine
import pandas as pd
from langchain_community.tools.sql_database.tool import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
import cachetools
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from cache import get_cached_result, cache_result


engine = create_engine('sqlite:///excel.db')

set_llm_cache(SQLiteCache(database_path=".cache.db"))
# set_llm_cache(InMemoryCache())
cache = cachetools.LRUCache(maxsize=1000)


llm = ChatOpenAI(
    base_url="https://api.sambanova.ai/v1/",
    api_key=os.environ.get('SAMBANOVA_API_KEY'),
    streaming=True,
    model="Meta-Llama-3.1-405B-Instruct",
)

prompt = '''You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
By default, use None for {top_k}, which means retrieve all rows without applying a LIMIT clause. If the user specifies a specific number of examples to obtain, use that value for {top_k} to limit the number of rows retrieved.
You can order the results to return the most informative data in the database.
Never query for all columns from a table; query only the columns needed to answer the question. Wrap each column name and table name in double quotes (") exactly as they appear in the database, without adding underscores or modifying spaces or capitalization.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Question: {input}
'''


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
def fetch_query_result(_engine, query):
    try:    
        sql_db = SQLDatabase(_engine)
        result = sql_db.run(query)
        return result
    except Exception as e:
        print(f"Something went wrong with fetch_query_result: {e}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
def cleanSQL(input):
    try:
        prompt = ChatPromptTemplate(
            messages=[
                ("system", "You are an expert SQL assistant. When a user inputs a flawed SQL query, your job is to identify the mistakes and provide a corrected version of the query. Wrap each column name and table name in double quotes exactly as they appear in the database, without adding underscores or modifying spaces or capitalization. The correction should maintain the same intent as the original query but follow proper syntax, logical conditions, and SQL best practices. Just reply with correct queries nothing else. Also remove ``` from the start and end of the queries."),
                ("user", "input: {input}")
            ]
        )
        
        cleanSQLChain = prompt | llm | StrOutputParser()
        cleaned_query = cleanSQLChain.invoke(input)
        return cleaned_query
    except Exception as e:
        print(f"Something went wrong with CleanSQLChain: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
def displayResult(sql, sqlResult):
    try:
        formatPrompt = ChatPromptTemplate(
            messages=[
                (
                    "system",
                    "You are an expert text editor. Your task is to present the results of a database query in a clear, concise, and professional format. Provide an explaination of the results but ensure that only essential and relevant information is included. Avoid unnecessary explanations or chatty language or any additional notes. The format should be clean, readable, and easy to interpret."
                ),
                (
                    "user",
                    "Query: {query}\nResults: {input}"
                )
            ]
        )
        
        chain3 = formatPrompt | llm | StrOutputParser()
        final_response = chain3.invoke({"query": sql, "input": sqlResult})
        return final_response
    except Exception as e: 
        print(f"Format failed with error: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
def fetchData(_engine, question):
    try:
        sql_query_prompt = PromptTemplate.from_template(prompt)
        sqlChain = create_sql_query_chain(llm=llm, db=SQLDatabase(_engine), prompt=sql_query_prompt)
        query = sqlChain.invoke({"question": question})   
        
        sqlQuery = cleanSQL(query)  
        print(sqlQuery)
        sqlResult = fetch_query_result(_engine, sqlQuery)
        result = displayResult(sqlQuery, sqlResult)
        return result
    except Exception as e:
        print(f"Something went wrong with sqlChain: {e}")
        raise
  
def process_question(engine, query):
    # Check if a similar query exists in the cache
    cached_result = get_cached_result(query)

    if cached_result:
        return cached_result
    else:
        new_result = fetchData(engine, query)
        
        # Cache the new result for future use
        cache_result(query, new_result)

        return new_result     

def main():
    st.set_page_config(page_title="Chat with your Excel")
    st.header("Chat with your Excel")
    # CSV file uploader in Streamlit
    csv_file = st.file_uploader("Upload your Excel file", type="xlsx")

    if csv_file is not None:
        excel_data = pd.ExcelFile(csv_file)

        for sheet_name in excel_data.sheet_names:
            df = pd.read_excel(csv_file, sheet_name=sheet_name)

            df.columns = df.columns.str.replace('\xa0', ' ')
            df.columns = df.columns.str.strip()

            df.to_sql(sheet_name, con=engine, index=False, if_exists='replace')

        user_question = st.text_input("Ask a question:")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                
                start_time = time.time()
                try:
                    response = process_question(engine, user_question)
                except Exception as e:
                    st.error(f"Something went wrong! Please try again. {e}")
                    return
                end_time = time.time()
                response_time = end_time - start_time 
                st.write(f"Time taken: {response_time:.2f} seconds")
                st.write(response)           
                
                
if __name__ == "__main__":
    main()
