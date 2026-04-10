from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool taht searches over the web for information.

    Args:
        query: The query to search for.

    Returns:
    The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)


llm = ChatOpenAI(model = "gpt-5")
tools = [search]

agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from search-agent")
    result = agent.invoke({"messages":HumanMessage("Search for 3 jobs using lanchain in Italy on linkedin and list their details")})
    print(result)

if __name__ == "__main__":
    main()

