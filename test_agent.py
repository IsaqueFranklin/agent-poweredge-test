import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents import create_react_agent

#Local llm config => poiting to gemma 4b.
llm = ChatOllama(model="gemma:4b")

#Defining the tools, in this case only the DuckDuckGo search.
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

#Creating the agent prompt. This is what tells the agent how it should behave, what tools it has, and how to use them.
#We use special placeholders that LangChain recognizes, such as: input, the user’s question, and agent_scratchpad, the short-term memory where the agent notes the steps it has already taken.
#Models with fewer parameters may get confused by large prompts.
prompt_template = """
Você é um assistente prestativo. Você tem acesso a uma ferramenta de busca.
Responda às perguntas do usuário. Se você não souber a resposta, use a ferramenta de busca.

Ferramentas disponíveis:
{tools}

Siga este formato para responder:

Pergunta: A pergunta do usuário
Raciocínio: O que você está pensando em fazer
Ferramenta: O nome da ferramenta a usar (ex: 'duckduckgo_search')
Entrada da Ferramenta: O que você vai perguntar à ferramenta
Observação: O resultado da ferramenta
... (este bloco de Raciocínio/Ferramenta/Entrada/Observação pode se repetir N vezes)
Raciocínio: Agora eu sei a resposta final
Resposta Final: A resposta final para o usuário

Comece!

Histórico da Conversa:
{chat_history}

Pergunta: {input}
Raciocínio:{agent_scratchpad}
"""

#(This template is based on hwchase17/react)
react_prompt_template = """
Responda às seguintes perguntas da melhor forma possível. Você tem acesso à seguinte ferramenta:

{tools}

Use o seguinte formato:

Pergunta: a pergunta de entrada que você deve responder
Raciocínio: você deve sempre pensar sobre o que fazer
Ferramenta: o nome da ferramenta a ser usada, deve ser uma de [{tool_names}]
Entrada da Ferramenta: a entrada para a ferramenta
Observação: o resultado da ferramenta
... (este Raciocínio/Ferramenta/Entrada da Ferramenta/Observação pode se repetir N vezes)
Raciocínio: agora eu sei a resposta final
Resposta Final: a resposta final para a pergunta original

Comece!

Pergunta: {input}
Raciocínio:{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(react_prompt_template)

#Creating the agent by defining the prompt and the tools it can use.
agent = create_react_agent(llm, tools, prompt=prompt)
#The create_react_agent function is designed for this type of prompt (reasoning/tool/input/observation).

#Creating the executor that will run the agent with the tools.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, #Important to see what the agent is doing.
    handle_parsing_errors=True #It helps to deal with parsing errors from gemma.
)

def format_chat_history(messages):
    if not messages:
        return "Nenhuma interação anterior."
    
    buffer = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            buffer.append(f"Usuário: {msg.content}")
        elif isinstance(msg, AIMessage):
            buffer.append(f"Agente: {msg.content}")
    return "\n".join(buffer)

#Conversation loop.
def main():
    #Memory list.
    chat_history = []

    print("Bem-vindo ao assistente de busca com Gemma 4B! \nDigite 'sair' para encerrar.")

    while True:
        try:
            user_input = input("\nVocê: ")

            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Encerrando a conversa. Até mais!")
                break
                
            #Formats the chat history into a string for the prompt.
            history_string = format_chat_history(chat_history)

            #Calls the agent executor with the user input and chat history.
            response = agent_executor.invoke({"input": user_input, "chat_history":history_string})

            output = response['output']

            print("---------------------------------")
            print(f"Agente: {output}")
            print("---------------------------------")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=output))

            #Holds only the last 10 interactions in memory to avoid prompt overflow.
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            print("Tentando novamente...\n")

if __name__ == "__main__":
    main()