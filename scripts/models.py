import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import Annotated, Sequence, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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


import functools

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display


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
            You are a recommender system.
            Recommend {n} items using the user's conversation.
            Sort the items by descending order of relevance.

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

    recommended_items = [
        item.strip().lower() for item in response["parsed"].items
    ]

    return recommended_items


async def recommend_agent(messages: List[BaseMessage], store: Any, n: int = 10):

    tavily_tool = TavilySearchResults(
        max_results=n,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        description="Search the web for information about books.",
    )

    books_retriever = create_retriever_tool(
        store.as_retriever(search_type="similarity", search_kwargs={"k": n * 2}),
        "retrieve_books",
        "Search and return books in the database.",
    )

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        recommended_items: Optional[List[str]] = None

    async def agent(state):

        messages = state["messages"]

        prompt = PromptTemplate.from_template(
            template="""
                    You are a conversational recommender system specialized in books.
                    Given the current conversation, recommend {n} books that align with the user's interests.
                    Every time you mention a book, consider its ID also.
                    Sort the items by descending order of relevance.

                    Current conversation is as follows:
                    {messages}
                """
        )

        llm = ChatOpenAI(
            temperature=0,
            streaming=True,
            model="gpt-4o",
            model_kwargs={"stream_options": {"include_usage": True}},
        )

        tools = [books_retriever, tavily_tool]

        llm = llm.bind_tools(tools)

        chain = prompt | llm

        response = await chain.ainvoke({"messages": messages, "n": n})

        return {"messages": [response]}

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

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(Recommendation, include_raw=True)
        formatter = prompt | structured_llm

        response = formatter.invoke({"message": last_message})

        recommended_items = [item.strip().lower() for item in response["parsed"].items]

        return {"recommended_items": recommended_items}

    def should_continue(state):

        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "tools"

    workflow = StateGraph(AgentState)
    tools = ToolNode([books_retriever, tavily_tool])

    workflow.add_node("agent", RunnableLambda(agent))
    workflow.add_node("tools", tools)
    workflow.add_node("formatter", formatter)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": "formatter"}  # "end": END,
    )
    workflow.add_edge("tools", "agent")

    workflow.add_edge("formatter", END)

    graph = workflow.compile()

    plot_graph(graph)

    state = {"messages": messages, "recommended_items": None}

    try:
        output = await graph.ainvoke(state)
    except Exception as e:
        output = {"messages": None, "recommended_items": None}

    return output


async def recommend_multi_agent(messages: List[BaseMessage], store: Any, n: int = 10):

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        sender: str
        recommended_items: Optional[List[str]] = None

    def create_agent(llm, tools, system_message: str):
        """Create an agent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | llm.bind_tools(tools)

    def agent_node(state, agent, name):
        result = agent.invoke(state)
        # We convert the agent output into a format that is suitable to append to the global state
        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            # Since we have a strict workflow, we can
            # track the sender so we know who to pass to next.
            "sender": name,
        }

    def router(state):
        # This is the router
        messages = state["messages"]
        
        last_message = messages[-1]
        second_last_message = messages[-2]

        if "FINAL ANSWER" in last_message.content or last_message.content.lower() == second_last_message.content.lower():
            # Any agent decided the work is done
            return "end"
        if last_message.tool_calls:
            # The previous agent is invoking a tool
            return "call_tool"
        return "continue"

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

        return {"recommended_items": recommended_items}

    llm = ChatOpenAI(model="gpt-4o")

    tavily_tool = TavilySearchResults(
        max_results=n,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        description="Search the web for information about books.",
    )

    books_retriever = create_retriever_tool(
        store.as_retriever(search_type="similarity", search_kwargs={"k": 2 * n}),
        "retrieve_books",
        "Search and return books in the database.",
    )

    research_agent = create_agent(
        llm,
        [tavily_tool],
        system_message="You should provide accurate data for the librarian_agent to use.",
    )
    research_node = functools.partial(
        agent_node, agent=research_agent, name="research_node"
    )

    librarian_agent = create_agent(
        llm,
        [books_retriever],
        system_message=f"You are a conversational recommender system specialized in books. Recommend {n} books that align with the user's interests.",
    )

    librarian_node = functools.partial(
        agent_node, agent=librarian_agent, name="librarian_node"
    )

    tools = [tavily_tool, books_retriever]
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("research_node", research_node)
    workflow.add_node("librarian_node", librarian_node)
    workflow.add_node("call_tool", tool_node)
    workflow.add_node("formatter", formatter)


    workflow.add_conditional_edges(
        "research_node",
        router,
        {
            "continue": "librarian_node",
            "call_tool": "call_tool",
            "end": "formatter",
        },
    )

    workflow.add_conditional_edges(
        "librarian_node",
        router,
        {
            "continue": "research_node",
            "call_tool": "call_tool",
            "end": "formatter",
        },
    )

    workflow.add_conditional_edges(
        "call_tool",
        # Each agent node updates the 'sender' field
        # the tool calling node does not, meaning
        # this edge will route back to the original agent
        # who invoked the tool
        lambda x: x["sender"],
        {
            "research_node": "research_node",
            "librarian_node": "librarian_node",
        },
    )

    workflow.add_edge(START, "research_node")
    workflow.add_edge("formatter", END)
    graph = workflow.compile()

    plot_graph(graph)

    state = {"messages": messages}
    output = await graph.ainvoke(state)

    return output
