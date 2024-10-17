from sqlalchemy import create_engine
import pandas as pd
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool,SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

llm = ChatOpenAI(
    base_url="https://api.sambanova.ai/v1/",
    api_key=os.environ.get('SAMBANOVA_API_KEY'),
    streaming=True,
    model="Meta-Llama-3.1-405B-Instruct",
    temperature=2
)

prompt = '''You are an expert SQL assistant. Your role is to provide precise and accurate SQL queries based strictly on the user's input. You must not add, change, or improvise any aspect of the query beyond what the user directly requests. You must not make assumptions or attempt to provide additional features or enhancements to the SQL queries. Simply generate the query as instructed, with no context modification or extra elements.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: {top_k}
Answer: "Final answer here"

Only use the following tables:

{table_info}.

Question: {input}'''

def main():
    st.set_page_config(page_title="Chat with your Excel")
    st.header("Chat with your Excel")

    # CSV file uploader in Streamlit
    csv_file = st.file_uploader("Upload your Excel file", type="xlsx")

    if csv_file is not None:
        engine = create_engine('sqlite:///my_excel_db.db')
        excel_data = pd.ExcelFile(csv_file)

        for sheet_name in excel_data.sheet_names:
            df = pd.read_excel(csv_file, sheet_name=sheet_name)

            df.columns = df.columns.str.replace('\xa0', ' ')
            df.columns = df.columns.str.strip()

            df.to_sql(sheet_name, con=engine, index=False, if_exists='replace')

        user_question = st.text_input("Ask a question:")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                sql_db = SQLDatabase(engine)
                sql_query_prompt = PromptTemplate.from_template(prompt)
                chain = create_sql_query_chain(llm=llm, db=sql_db, prompt=sql_query_prompt)
                query = chain.invoke({"question": user_question})


                clean_sql = ChatPromptTemplate(
                    messages=[
                        ("system", "You are an expert SQL assistant. When a user inputs a flawed SQL query, your job is to identify the mistakes and provide a corrected version of the query. The correction should maintain the same intent as the original query but follow proper syntax, logical conditions, and SQL best practices. Just reply with correct queries noting else. Also remove ``` from the start and end of the queries and LIMIT clause from the queries response"),
                        ("user", "input: {input}")
                    ]
                )
                


                output_parser = StrOutputParser()
                chain2 = clean_sql | llm | output_parser
                cleaned_query = chain2.invoke(query)
                print(cleaned_query)
                
                result = sql_db.run(cleaned_query)


                format_result = ChatPromptTemplate(
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

                output_parser = StrOutputParser()
                chain3 = format_result | llm | output_parser
                final_response = chain3.invoke({"query": cleaned_query, "input": result})
                st.write(final_response)
                
                
                
                
if __name__ == "__main__":
    main()
