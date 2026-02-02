from langchain.tools import tool
from utils import vectorstore

@tool
def search_document(query: str) -> str:
    """
    Retrieve information to help answer a query.
    Returns the answer to the query.
    """
    docs = vectorstore.similarity_search(query, k=3)

    # retriever = vectorstore.as_retriever()
    # docs = retriever.invoke(query)

    context = '\n\n'.join([doc.page_content for doc in docs])
    return context

tools = [search_document]

# https://docs.langchain.com/oss/python/langgraph/agentic-rag
# Check the retriever tool