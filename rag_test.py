import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

import load_dotenv

# Load environment variables from .env file
load_dotenv.load_dotenv()
mytoken = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Step 1: Prepare your website content
# For testing purposes, let's create a sample text based on your CV
sample_website_content = """
Saksonita Khoeurn
Ph.D. | Data Scientist | Startup Enthusiast

About Me
Data Scientist with a Ph.D. in Engineering, blending machine learning expertise with business acumen.

Research Focus
Predictive maintenance and explainable AI for semiconductor manufacturing and healthcare applications.

Technical Expertise
Developing AI models and data analytics solutions at Big Data Labs for complex business problems.

Entrepreneurial Drive
Mentored startups and designed accelerator programs at Techo Startup Center in Cambodia.

Knowledge Sharing
Former lecturer and IT instructor passionate about teaching data science and development.

"Creating solutions that bridge cutting-edge research with practical business applications."

Work Experience
Data Scientist
Big Datalabs | July 2024 - Present
- Leveraging advanced data analytics and machine learning techniques to drive insights.
- Developing predictive models and optimizing business processes.
- Contributing to data-driven decision-making within the organization.

Startup Development Specialist
Techo Startup Center, Cambodia | 2020 - 2023
- Led and organized national startup programs and accelerator initiatives.
- Mentored startups and facilitated connections with relevant industry mentors.

University Lecturer
Norton University, Cambodia | 2018 - 2019
- Taught courses in Web Application Development and MS SQL Server.
- Guided students through technical concepts and practical applications.

Assistant Digital Marketing Manager
Westec Media Limited, Cambodia | 2016
- Managed client relationships and coordinated project resources.
- Planned and executed marketing campaigns with a focus on analytics.

Information Communication Officer
NGO Education Partnership, Cambodia | 2015
- Developed content and managed communications for newsletters and events.
- Resolved IT and communication issues for staff and members.

IT Instructor
Korea Software HRD Center, Cambodia | 2015 - 2016
- Taught IT courses and mentored students in academic and personal growth.
- Designed and implemented lesson plans and assessments.

Education
Ph.D in Engineering
Chungbuk National University, South Korea
2021 - 2025 | GPA 4.23/4.5

Master's Degree in ICT
Handong Global University, South Korea
2018 - 2020 | Best Supporter Award

Bachelor's Degree in Computer Science
Norton University, Cambodia
2015 - 2018 | Ranked No. 1 in senior year

Skills & Certificates
Technical Skills
- Python (Expert)
- Predictive Analytics (Expert)
- Data Analytics (Expert)
- Descriptive Analytics (Expert)
- PyTorch (Advanced)
- TensorFlow (Advanced)
- SQL (Proficient)

Language Skills
- English (Professional Working Proficiency)
- Korean (TOPIK Level 5)
- Advanced level in reading, writing, speaking, and listening

Certificates
- AI Agents Fundamentals - Hugging Face (May 2023)
- Neural Networks and Deep Learning - Coursera (November 2023)
- Cambridge FinTech and Regulatory Innovation - Cambridge Judge Business School (March 2021)
- GET 2016 UNITWIN Global Education and Training Program (2016)

Publications
- Improving the Reusability of Patient Controlled Analgesia Data based on Data Products and Experimental Implementation (2024)
- Development and Verification of an AI Model for Melon Import Prediction (2023)
- A Comparison of Time Series Forecast Models for Predicting the Outliers Particles in Semiconductor Cleanroom (2022)
- Barriers to E-commerce Business Model in Cambodia: A Case Study
- Sentiment Analysis Engine for Cambodian Music Industry Rebuilding
- AKN: Cambodia News Integration Service (2018)
"""

# Step 2: Process the content and create documents
def create_documents_from_text(text):
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Group lines into meaningful chunks (sections)
    sections = []
    current_section = ''
    
    for line in lines:
        if line.strip() == '':
            if current_section:
                sections.append(current_section)
                current_section = ''
        else:
            if not current_section:
                current_section = line
            else:
                current_section += '\n' + line
    
    if current_section:
        sections.append(current_section)
    
    # Create documents from sections
    documents = []
    for section in sections:
        documents.append(Document(page_content=section))
    
    return documents

# Step 3: Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_documents(documents)

# Step 4: Create vector store
def create_vector_store(chunks):
    # Use a small, efficient embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create the vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

# Step 5: Set up a Hugging Face Hub model (completely CPU compatible)
def setup_cpu_friendly_llm():
    # Use Hugging Face Hub with a smaller model
    # You'll need a HuggingFace token for this
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = mytoken # Replace with your token
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # CPU-friendly model, 250M parameters
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    
    return llm

# Step 6: Create RAG chain
def create_rag_chain(vector_store, llm):
    # Create a custom prompt template
    template = """
You are an AI assistant for Saksonita Khoeurn, a Data Scientist with a Ph.D. in Engineering. 
You help answer questions about Saksonita's background, experience, skills, and publications.
Please be professional, concise, and accurate in your responses.

Context information is below:
-----------------
{context}
-----------------

Given the context information and not prior knowledge, answer the question: {question}
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create the retrieval chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain

# Step 7: Main function to initialize everything
def initialize_rag_agent():
    # Process website content
    documents = create_documents_from_text(sample_website_content)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    
    # Set up CPU-friendly LLM
    llm = setup_cpu_friendly_llm()
    
    # Create RAG chain
    rag_chain = create_rag_chain(vector_store, llm)
    
    return rag_chain

# Step 8: Function to query the RAG agent
def query_rag_agent(rag_chain, query):
    response = rag_chain({"query": query})
    return response["result"]

# Example usage
if __name__ == "__main__":
    # Initialize the RAG agent
    print("Initializing RAG agent...")
    rag_chain = initialize_rag_agent()
    
    # Example queries
    queries = [
        "What is Saksonita's educational background?",
        "What are Saksonita's research interests?",
        "What publications has Saksonita written?",
        "What technical skills does Saksonita have?",
        "What was Saksonita's role at Techo Startup Center?"
    ]
    
    # Process queries
    for query in queries:
        print(f"\nQuestion: {query}")
        answer = query_rag_agent(rag_chain, query)
        print(f"Answer: {answer}")