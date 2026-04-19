import streamlit as st
import os
from PyPDF2 import PdfReader

# استدعاءات حديثة ومستقرة تماماً
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

# --- إعدادات الأمان ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except:
    openai_api_key = None

# --- وظيفة معالجة الملفات ---
def load_legal_docs(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.pdf'):
                path = os.path.join(folder_path, filename)
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
    return documents

# --- الواجهة ---
st.set_page_config(page_title="المستشار القانوني للجماعات", layout="wide")
st.title("⚖️ منصة الذكاء الاصطناعي للقوانين الجماعية")

lib_path = "laws_library"

with st.sidebar:
    st.header("⚙️ الإعدادات")
    if st.button("🚀 تحديث وفهرسة القوانين"):
        with st.spinner("جاري معالجة القوانين..."):
            raw_docs = load_legal_docs(lib_path)
            if raw_docs and openai_api_key:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(raw_docs)
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local("legal_vector_db")
                st.success("تم التحديث بنجاح!")
            else:
                st.error("تأكد من وجود الملفات ومفتاح API.")

# --- محرك البحث الحديث ---
query = st.text_input("🔍 اسأل عن أي مقتضى قانوني:")

if query and openai_api_key:
    if os.path.exists("legal_vector_db"):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.load_local("legal_vector_db", embeddings, allow_dangerous_deserialization=True)
        
        # 1. استرجاع النصوص ذات الصلة
        retrieved_docs = db.similarity_search(query, k=3)
        
        # 2. إعداد القالب (Prompt)
        prompt = ChatPromptTemplate.from_template("""
        أجب على السؤال بناءً على النصوص القانونية المقدمة فقط:
        <context>
        {context}
        </context>
        السؤال: {input}""")

        # 3. إنشاء السلسلة الحديثة
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # 4. توليد الإجابة
        response = document_chain.invoke({"input": query, "context": retrieved_docs})
        
        st.markdown("### 📝 الإجابة القانونية:")
        st.info(response)
        
        st.markdown("---")
        st.subheader("📚 المصادر:")
        for doc in retrieved_docs:
            st.write(f"🔹 {doc.metadata['source']} (ص {doc.metadata['page']})")
