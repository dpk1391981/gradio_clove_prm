from langchain.vectorstores.cassandra import Cassandra
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document
from src.helper import download_huggingface_embedding
from src.prompt import sytem_prompt,contextual_q_system_prompt

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    
    embeddings = download_huggingface_embedding()
    
    astra_vector_store = Cassandra(
        embedding=embeddings,
        session=None,
        keyspace=state["astraConfig"]["keyspace"],
        table_name=state["astraConfig"]["table"]
    )

    retriever = astra_vector_store.as_retriever()
    
    
    #history retriever prompt
    contextual_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system", contextual_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    history_aware_retriver=create_history_aware_retriever(state["llm"], retriever,contextual_q_prompt)
    
    #Answer
    qa_prompt=ChatPromptTemplate.from_messages(
        [
            ("system", sytem_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    
    question_answer_chain=create_stuff_documents_chain(state["llm"],qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriver,question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            state['get_session_history'],
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            callbacks=[state["callbacks"]]
        )
    
    question = state["question"]
    
    documents = conversational_rag_chain.invoke(
                {"input": question},
                config={
                    "configurable": {"session_id": state["session_id"]}
                },
            )
    
  

    # Accessing components
    answer = documents["answer"]
    query_result = Document(page_content=answer)
    return {"documents": query_result, "question": question}