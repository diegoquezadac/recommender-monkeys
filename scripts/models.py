import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import Annotated, Sequence, Optional
from typing_extensions import TypedDict
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

import functools

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

from langgraph.prebuilt import create_react_agent


def plot_graph(graph):
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass


def _get_inference_cost(messages: list[BaseMessage]) -> float:
    input_tokens = 0
    output_tokens = 0

    for message in messages:
        if isinstance(message, AIMessage):
            input_tokens += message.usage_metadata["input_tokens"]
            output_tokens += message.usage_metadata["output_tokens"]
    return input_tokens * 2.5 / 1_000_000 + output_tokens * 10 / 1_000_000


def recommend_random(items: List[str], n: int = 5):
    return np.random.choice(items, n, replace=False).tolist()


def recommend_most_popular(item_interactions: Dict[str, int], n: int = 5):
    most_popular_items = sorted(
        item_interactions, key=item_interactions.get, reverse=True
    )[:n]
    return most_popular_items


async def recommend_zero_shot(
    messages: List[BaseMessage],
    n: int = 5,
    verbose: bool = False,
):

    class Recommendation(BaseModel):
        items: List[str] = Field(
            description="A list of items names recommended by the system. Do not include the author's name or any other metadata."
        )

    system = """
            You are a conversational recommender system specialized in books.
            Given the current conversation, recommend {n} books that align with the user's interests.
            Sort the books by descending order of relevance.

            Conversation:
            {messages}
        """

    prompt = ChatPromptTemplate.from_messages([("system", system)])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(Recommendation, include_raw=True)

    zero_shot_structured_llm = prompt | structured_llm
    response = await zero_shot_structured_llm.ainvoke({"messages": messages, "n": n})

    prompt_tokens = response["raw"].response_metadata["token_usage"]["prompt_tokens"]
    completion_tokens = response["raw"].response_metadata["token_usage"][
        "completion_tokens"
    ]

    cost = 2.5 * prompt_tokens / 1000000 + 10 * completion_tokens / 1000000

    if verbose:
        print(f"Prompt Tokens: {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        print(f"Cost: ${cost} USD")
        print(f"Recommended Items: {response['parsed'].items}")

    recommended_items = [item.strip().lower() for item in response["parsed"].items]

    return recommended_items


async def recommend_agent_2(messages: List[BaseMessage], store: Any, n: int = 10):

    system = f"""
        You are a recommender system specialized in books.
        Given the current conversation, you must recommend {n} books that align with the user's interests.
        Every time you mention a book, consider its ID also.
        Sort the books by descending order of relevance.
        You must not ask any questions to the user, simply return the recommendations.
        """

    llm = ChatOpenAI(model="gpt-4o")

    books_retriever = create_retriever_tool(
        store.as_retriever(search_type="similarity", search_kwargs={"k": 2 * n}),
        "retrieve_books",
        "Search books in the database.",
    )

    librarian_agent = create_react_agent(
        llm,
        tools=[books_retriever],
        state_modifier=system,
    )

    def formatter(state):

        messages = state["messages"]
        last_message = messages[-1]

        class Recommendation(BaseModel):
            items: List[str] = Field(
                description="A list of items ids retrieved from the database."
            )

        system = """
            Format the following message as a recommendation: {message}
            """

        prompt = ChatPromptTemplate.from_messages([("system", system)])

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = llm.with_structured_output(Recommendation, include_raw=True)
        formatter = prompt | structured_llm

        response = formatter.invoke({"message": last_message})

        recommended_items = [item.strip().lower() for item in response["parsed"].items]

        messages.append(AIMessage(content=recommended_items, name="formatter"))

        return {"messages": messages}

    def librarian_node(state: MessagesState) -> MessagesState:
        result = librarian_agent.invoke(state)
        result["messages"][-1] = AIMessage(
            content=result["messages"][-1].content, name="librarian"
        )
        return {
            "messages": result["messages"],
        }

    workflow = StateGraph(MessagesState)
    workflow.add_node("formatter", formatter)
    workflow.add_node("librarian", librarian_node)
    workflow.add_edge("librarian", "formatter")
    workflow.add_edge("formatter", END)
    workflow.add_edge(START, "librarian")
    graph = workflow.compile()

    plot_graph(graph)

    state = {"messages": messages}

    try:
        output = await graph.ainvoke(state)
    except Exception as e:
        output = {"messages": None}

    return output


async def recommend_multi_agent_2(messages: List[BaseMessage], store: Any, n: int = 10):

    def make_system_prompt(suffix: str) -> str:
        return (
        " You are a helpful AI assistant, collaborating with other assistants."
       f" Your task is to help the team recommend {n} books to the user."
        " Every time you mention a book, consider its ID also."
        " Sort the final recommendation by descending order of relevance."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

    llm = ChatOpenAI(model="gpt-4o")

    tavily_tool = TavilySearchResults(
        max_results=n//2,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        description="Search the web for information about books.",
        name="search_book_info"
    )

    books_retriever = create_retriever_tool(
        store.as_retriever(search_type="similarity", search_kwargs={"k": 2 * n}),
        "retrieve_books",
        "Search and return books in the database.",
    )

    researcher_agent = create_react_agent(
        llm,
        tools=[tavily_tool],
        state_modifier="You can only do research. You are working with a librarian colleague.",
    )

    def formatter(state):

        messages = state["messages"]
        last_message = messages[-1]

        class Recommendation(BaseModel):
            items: List[str] = Field(
                description="A list of items ids recommended by the system."
            )

        system = """
            Format the following message as a recommendation: {message}
            """

        prompt = ChatPromptTemplate.from_messages([("system", system)])

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = llm.with_structured_output(Recommendation, include_raw=True)
        formatter = prompt | structured_llm

        response = formatter.invoke({"message": last_message})

        recommended_items = [item.strip().lower() for item in response["parsed"].items]

        messages.append(AIMessage(content=recommended_items, name="formatter"))

        return {"messages": messages}

    def researcher_node(state: MessagesState) -> MessagesState:
        result = researcher_agent.invoke(state)
        # wrap in a human message, as not all providers allow
        # AI message at the last position of the input messages list
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="researcher"
        )
        return {
            "messages": result["messages"],
        }

    librarian_agent = create_react_agent(
        llm,
        [books_retriever],
        state_modifier=make_system_prompt(
            "You can only recommend books from your catalog. You are working with a researcher colleague."
        ),
    )

    def librarian_node(state: MessagesState) -> MessagesState:
        result = librarian_agent.invoke(state)
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="librarian"
        )
        return {
            "messages": result["messages"],
        }

    def router(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if "FINAL ANSWER" in last_message.content:
            return END
        return "continue"

    workflow = StateGraph(MessagesState)
    workflow.add_node("formatter", formatter)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("librarian", librarian_node)

    workflow.add_conditional_edges(
        "librarian",
        router,
        {"continue": "researcher", END: "formatter"},
    )

    workflow.add_edge("researcher", "librarian")

    workflow.add_edge("formatter", END)
    workflow.add_edge(START, "researcher")
    graph = workflow.compile()

    plot_graph(graph)

    state = {"messages": messages}

    try:
        output = await graph.ainvoke(state)
    except Exception as e:
        output = {"messages": None}

    return output
