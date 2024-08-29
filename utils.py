from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import git
import os
import deeplake
from queue import Queue
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
local = True
if local:
    from dotenv import load_dotenv
    load_dotenv()
from langchain.chains import conversational_retrieval
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
allowed_extensions = ['.py', '.ipynb', '.md','.pdf']
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set it in the .env file.")

genai.configure(api_key=api_key)


# from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings



class Embedder:
    def __init__(self, git_link) -> None:
        self.git_link = git_link
        last_name = self.git_link.split('/')[-1]
        self.clone_path = last_name.split('.')[0]
        self.deeplake_path = f"hub://manavg23/{self.clone_path}"
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # switch to 'gpt-4'
        self.hf = HuggingFaceEmbeddings(model_name=model_name)
        self.openai = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="GEMINI_API_KEY")
        self.MyQueue =  Queue(maxsize=5)

    def add_to_queue(self, value):
        if self.MyQueue.full():
            self.MyQueue.get()
        self.MyQueue.put(value)

    def clone_repo(self):
        if not os.path.exists(self.clone_path):
            # Clone the repository
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try: 
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e: 
                        pass
    
    def chunk_files(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)
        self.num_texts = len(self.texts)

    def embed_deeplake(self):
        # db = DeepLake(dataset_path=self.deeplake_path, embedding_function= OpenAIEmbeddings())
        db = DeepLake(dataset_path=self.deeplake_path, embedding_function= self.hf)
        db.add_documents(self.texts)
        ## Remove data from the cloned path
        self.delete_directory(self.clone_path)
        return db
    
    def delete_directory(self, path):
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path)
        
    def load_db(self):
        exists = deeplake.exists(self.deeplake_path)
        if exists:
            ## Just load the DB
            self.db = DeepLake(
            dataset_path=self.deeplake_path,
            read_only=True,
            embedding_function=self.hf,
             )
        else:
            ## Create and load
            self.extract_all_files()
            self.chunk_files()
            self.db = self.embed_deeplake()

        self.retriever = self.db.as_retriever()
        self.retriever.search_kwargs['distance_metric'] = 'cos'
        self.retriever.search_kwargs['fetch_k'] = 100
        self.retriever.search_kwargs['k'] = 3


    def retrieve_results(self, query):
        chat_history = list(self.MyQueue.queue)
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
        history_aware_retriever = create_history_aware_retriever(
    llm,self.retriever ,contextualize_q_prompt
)
        qa_system_prompt = """You are an assistant for question-answering tasks. \
                        Use the following pieces of retrieved context to answer the question. \
                If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\
            {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        
            


        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        self.add_to_queue((query, result["answer"]))
        return result['answer']
