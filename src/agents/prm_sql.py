from langchain.prompts import SemanticSimilarityExampleSelector, PromptTemplate, FewShotPromptTemplate
from src.helper import download_huggingface_embedding, few_shots
from src.prompt import mysql_prompt
from langchain.vectorstores import FAISS
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from src.dbconfig import config_mysql_db
from langchain.agents.agent_types import AgentType
from langchain.schema import Document


embeddings = download_huggingface_embedding()
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=few_shots(),
    embeddings=embeddings,
    vectorstore_cls=FAISS,
    k=2,
)

example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"],  # These variables are used in the prefix and suffix
)


def sql_agent(state):
    print("---sql agent---")
    question = state["question"]
    print(question)
    print(state)

    db = config_mysql_db(
        state["dbconfig"]["host"],
        state["dbconfig"]["user"],
        state["dbconfig"]["pass"],
        state["dbconfig"]["db_name"],
    )

    toolkit = SQLDatabaseToolkit(llm=state["llm"], db=db)

    limited_query = f"{question} LIMIT {state['dbconfig']['limit']}"

    # sql agent
    agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=state["llm"],
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,  # Enable parsing error handling
        prompt=few_shot_prompt,  # Use the FewShotPromptTemplate
    )

    docs = agent.run(limited_query, callbacks=[state["callbacks"]])
    query_result = Document(page_content=docs)

    return {"documents": query_result, "question": question}
