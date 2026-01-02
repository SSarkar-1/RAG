from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

load_dotenv()

persistent_directory="db/chroma_db"
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
db=Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

# Set up the AI model
model=ChatOpenAI(model="gpt-4o")

# Store our conservation as messages
chat_history=[]

def ask_question(user_question):
    print(f"--- You asked :{user_question} ---")

    #Step 1: Make the history clear using conversation history
    if chat_history:
        # Ask AI to make question standalone 
        messages=[
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [HumanMessage(content=f"New question : {user_question}")]

        result=model.invoke(messages)
        search_question=result.content.strip()
        print(f"Searching for :{search_question}")

    else:
        search_question=user_question

    # Step 2: Find relevant documents
    retriver=db.as_retriever(search_kwargs={"k":3})
    docs=retriver.invoke(search_question)

    print(f"Found relevant documents")
    for i,doc in enumerate(docs,1):
        # Show first 2 lines of each document 
        lines=doc.page_content.split('\n')[:2]
        preview='\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Create final prompt
    combined_input=f"""Based on the following documents, please answer this question: {search_question}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents." """

    #Step 4: Get the answer
    messages=[SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),]+chat_history+[HumanMessage(content=combined_input)]

    result=model.invoke(messages)
    answer=result.content

    # Step 5: Remember This conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit")
    while True:
        question=input("\nYour Question : ")

        if question.lower()=='quit':
            print("Goodbye")
            break
        ask_question(question)

if __name__=="__main__":
    start_chat()