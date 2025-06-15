import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain,LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate

st.title("Text to Math Problem Solver")
groq_api_key = st.sidebar.text_input(label="Groq Api Key",type="password")

if not groq_api_key:
    st.info("Please add your Groq API key")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It",api_key=groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tools for searching the Internet to find various info on the topic"
)

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering maths related questions. Only math input needed else give error"
)

prompt = """
You are agent tasked for solving users mathematical question. Logically arrive at the solution and provide a detailed explanation
and display iy point wise for the question below
Question: {question}
Answer:  
"""

prompt_template = PromptTemplate(
    input_variables=['question'],
    template=prompt
)

chain = LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for for answering logic-based and reasoning questions"
)

assistant_agent = initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi, I'm a Math chatbot that can answer all your questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def generate_response(user_question):
    response = assistant_agent.invoke({"input":user_question})
    return response

question = st.text_area(label="Enter your question",value="I have 5 bananas and 7 grapes. I eat 2 bananas give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contain 25 blueberries. How many total pieces of fruits do I have in the end?")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response"):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(parent_container=st.container(),expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("### Response")
            st.success(response)
    else:
        st.warning("Please enter the question")
