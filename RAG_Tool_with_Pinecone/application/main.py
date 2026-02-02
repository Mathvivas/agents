from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from application.workflow import app

SYSTEM_PROMPT = '''
You are a helpful AI assistant with access to one search tool:

1. search_document: Use this to search the document for something related to the user's question.
Example: What is Data Ingestion? --> Use search_document to search the document for the answer.

Answer only using the search_document, do not use your own knowledge.
If you CAN'T find the answer in the document, just say you couldn't find the answer.
'''

def main():
    print('Agent Chat (type "quit", "exit" or "q" to end)')
    print('-' * 50)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_input = input('\nYou: ').strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print('Goodbye!')
            break

        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))

        try:
            print('\nAgent: ', end='', flush=True)

            for event in app.stream({'messages': messages}):
                for node_name, node_output in event.items():
                    if node_name == 'agent':
                        last_msg = node_output['messages'][-1]

                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            print(last_msg.content, end='', flush=True)
                    elif node_name == 'tools':
                        print('\n[Using tools...]', end='', flush=True)

            print()

            final_state = app.invoke({'messages': messages})
            messages = final_state['messages']

        except Exception as e:
            print(f'\Error: {str(e)}')
            print('Please try again.')

main()