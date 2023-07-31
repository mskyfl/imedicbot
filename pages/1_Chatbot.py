import streamlit as st
import os
import time
import datetime
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.callbacks import get_openai_callback

# Setting up Streamlit page configuration
st.set_page_config(
    layout="centered", 
    initial_sidebar_state="expanded"
)


# Getting the OpenAI API key from Streamlit Secrets
openai_api_key = st.secrets.secrets.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# Getting the Pinecone API key and environment from Streamlit Secrets
PINECONE_API_KEY = st.secrets.secrets.PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = st.secrets.secrets.PINECONE_ENV
os.environ["PINECONE_ENV"] = PINECONE_ENV
# Initialize Pinecone with API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


param1 = True
@st.cache_data
def select_index(__embeddings):
    if param1:
        pinecone_index_list = pinecone.list_indexes()
    return pinecone_index_list


# Set the text field for embeddings
text_field = "text"
# Create OpenAI embeddings
embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
model_name = st.sidebar.selectbox(label="Select Model", options=MODEL_OPTIONS)
lang_options = ["English", "German", "French", "Chinese", "Italian", "Japanese", "Arabic", "Hindi", "Turkish", "Urdu"]
lang_dic = {"English":"\nAnswer in English", "German":"\nAnswer in German", "French":"\nAnswer in French", "Chinese":"\nAnswer in Chinese", "Italian":"\nAnswer in Italian", "Japanese":"\nAnswer in Japanese", "Arabic":"\nAnswer in Arabic", "Hindi":"\nAnswer in Hindi", "Turkish":"\nAnswer in Turkish", "Urdu":"\nAnswer in Urdu"}
language = st.sidebar.selectbox(label="Select Language", options=lang_options)

@st.cache_resource
def ret(pinecone_index):
    if pinecone_index != "":
        # load a Pinecone index
        index = pinecone.Index(pinecone_index)
        time.sleep(5)
        db = Pinecone(index, embeddings.embed_query, text_field)
    return db

@st.cache_resource
def init_memory():
    return ConversationSummaryMemory(llm = ChatOpenAI(model_name = model_name),
                                           memory_key="chat_history", 
                                           return_messages=True,
                                           max_token_limit=200, 
                                           verbose=True)
memory = init_memory()

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
standalone question without changing the content in given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
condense_question_prompt_template = PromptTemplate.from_template(_template)

prompt_template = """You are helpful information giving QA System and make sure you don't answer anything 
not related to following context. You are always provide useful information & details available in the given context. Use the following pieces of context to provide detailed and informative answer to the question at the end. 
Also check chat history if question can be answered from it or question asked about previous history. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer should be long and detailed.

{context}
Chat History: {chat_history}
Question: {question}
"""
pt = lang_dic[language]
#st.sidebar.write(prompt_template+pt)
prompt_temp = prompt_template + pt
qa_prompt = PromptTemplate(
    template=prompt_temp, input_variables=["context", "chat_history","question"]
)
# PROMPT.format(language=language)
# chain_type_kwargs = {"prompt": PROMPT}

pinecone_index_list = select_index(embeddings)
pinecone_index = st.sidebar.selectbox(label="Select Index", options = pinecone_index_list )

templat = """You are helpful information giving QA System and make sure you don't answer anything not related to following context. You are always provide useful information & details available in the given context. Use the context to provide long, detailed and informative answer to the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer should be long and detailed.


Chat History: {chat_history}
Question: {human_input}
Use following context to answer the Question.
Context:"""

# template = template + pt
def chat(pinecone_index):

    db = ret(pinecone_index)
    @st.cache_resource
    def agent_meth(query, pt):
        search = DuckDuckGoSearchRun()
        web_res = search.run(query)
        #doc_res = db.similarity_search(query)
        #result_string = ' '.join(stri.page_content for stri in doc_res)
        contex = "\n " + web_res + "\nAssistant:" + pt #+ result_string
        templ = templat + contex
        promptt = PromptTemplate(input_variables=["chat_history", "human_input"], template=templ)
        agent = LLMChain(
            llm=OpenAI(model_name = model_name, temperature=0),
            prompt=promptt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=2, 
                                                  memory_key="chat_history", 
                                                )
                                                  
        )
        
        
        return agent, contex
    @st.cache_resource
    def retr(prompt_temp):
        llm = ChatOpenAI(model_name = model_name, temperature=0.1)
        question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory, verbose=True)
        doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt, verbose=True)
        agent = ConversationalRetrievalChain(
            retriever=db.as_retriever(search_kwargs={'k': 6}),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            memory=memory,
            verbose=True
        )

        return agent


    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model_name

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    #agent = conversational_chat()
    st.sidebar.write("---")
    st.sidebar.write("Enable Web Access (Excludes Document Access)")
    meth_sw = st.sidebar.checkbox("Web Search")
    st.sidebar.write("---")
    #chat_history = []
    if prompt := st.chat_input():
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content":prompt})
        # st.chat_message("user").write(prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            st_callback = StreamlitCallbackHandler(st.container())
            if meth_sw:
                agent, context = agent_meth(prompt, pt)
                #st.sidebar.write(agent.agent.llm_chain.prompt.template)
                response = agent.predict(human_input=prompt, callbacks=[st_callback])#.run(prompt, callbacks=[st_callback])
                st.session_state.chat_history.append((prompt, response))
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                agent = retr(prompt_temp)
                with st.spinner("Thinking..."):
                    with get_openai_callback() as cb:
                        response = agent({'question': prompt, 'chat_history': st.session_state.chat_history})#.run(prompt)#, callbacks=[st_callback])
                        #st.write(response)
                        st.session_state.chat_history.append((prompt, response['answer']))
                        message_placeholder.markdown(response['answer'])
                        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                st.sidebar.header("Total Token Usage:")
                st.sidebar.write(f"""
                        <div style="text-align: left;">
                            <h3>   {cb.total_tokens}</h3>
                        </div> """, unsafe_allow_html=True)
if pinecone_index != "":
    chat(pinecone_index)
    #st.sidebar.write(st.session_state.messages)
    #don_check = st.sidebar.button("Download Conversation")
    con_check = st.sidebar.button("Upload Conversation to loaded Index")
    
    text = []
    for item in st.session_state.messages:
        text.append(f"Role: {item['role']}, Content: {item['content']}\n")
    #st.sidebar.write(text)
    if con_check:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(text)
        st.sidebar.info('Initializing Conversation Uploading to DB...')
        time.sleep(11)
        # Upload documents to the Pinecone index
        vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
        
        # Display success message
        st.sidebar.success("Conversation Uploaded Successfully!")
    
    text = '\n'.join(text)
    # Provide download link for text file
    st.sidebar.download_button(
        label="Download Conversation",
        data=text,
        file_name=f"Conversation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
        mime="text/plain"
    )
