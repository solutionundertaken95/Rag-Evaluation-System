import tools as tl
from prompts import system_agent_prompt
from tools import collecting_info
from objects import Answer
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from objects import Answer

llm = init_chat_model(model="gemini-3.6-flash", temperature=0, model_provider="google-genai")

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt,
    response_format=Answer
)

def ask(question) -> dict:
    """ Answer the user's question about the pdf"""

    tl.last_retrieved_context = []

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    structured_response = result["structured_response"]

    return {
        "answer" : structured_response.answer,
        "context": structured_response.context
    } 

print(ask("what is the diet of Penguins"))