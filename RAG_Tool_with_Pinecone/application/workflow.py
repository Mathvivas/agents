from utils import should_continue_node, agent_node, AgentState
from tools import tools
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

workflow = StateGraph(AgentState)

workflow.add_node('agent', agent_node)
workflow.add_node('tools', ToolNode(tools))

workflow.set_entry_point('agent')

workflow.add_conditional_edges('agent',
                               should_continue_node,
                               {'CONTINUE': 'tools', 'END': END}
                               )

workflow.add_edge('tools', 'agent')

app = workflow.compile()