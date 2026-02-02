from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from tools import tools
from utils import llm

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent_node(state: AgentState):
    '''The agent node that decides what to do next.'''
    messages = state['messages']
    response = llm.bind_tools(tools).invoke(messages)
    return {'messages': [response]}

def should_continue_node(state: AgentState):
    '''Decides if the agent should continue or end.'''
    messages = state['messages']
    last_message = messages[-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return 'END'
    return 'CONTINUE'

