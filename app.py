import streamlit as st
import os
import zipfile
from PyPDF2 import PdfReader

# استدعاءات Google Gemini و LangChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المستشار القانوني للجماعات (Gemini)", layout="wide")
st.title("⚖️ منصة الذكاء الاصطناعي للقوانين (مدعوم من Gemini)")

# --- إعدادات مفتاح API ---
# احصل على مفتاحك من: https://aistudio.google.com/
gemini_api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not gemini_api_key:
    st.info("⚠️ يرجى إضافة مفتاح Gemini API للبدء. يمكنك الحصول عليه مجاناً من Google AI Studio.")
    st.stop()

# --- وظائف معالجة الملفات ---
def handle_zip_file(zip_path, extract_to="laws_library"):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    return False

def load_legal_docs(folder_path):
    documents = []
    if not os.path.exists(folder_path): return None
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for filename in files:
        path = os.path.join(folder_path, filename)
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
        except: continue
    return documents

# --- القائمة الجانبية ---
with st.sidebar:
    st.header("⚙️ الإعدادات")
    if os.path.exists("laws_library.zip"):
        handle_zip_file("laws_library.zip")
    
    uploaded_files = st.file_uploader("ارفع ملفات PDF:", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 بدء الفهرسة (Gemini)"):
        all_docs = []
        if uploaded_files:
            for f in uploaded_files:
                reader = PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text: all_docs.append(Document(page_content=text, metadata={"source": f.name, "page": i + 1}))
        
        folder_docs = load_legal_docs("laws_library")
        if folder_docs: all_docs.extend(folder_docs)

        if all_docs:
            with st.spinner("جاري التحليل باستخدام ذكاء Google..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = splitter.split_documents(all_docs)
                
                # استخدام Embeddings الخاصة بـ Google
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local("legal_vector_db_gemini")
                st.success("✅ تمت الفهرسة بنجاح مجاناً!")
        else:
            st.error("لا توجد ملفات!")

# --- محرك البحث ---
query = st.text_input("🔍 اسأل عن أي مقتضى قانوني:")

if query:
    if os.path.exists("legal_vector_db_gemini"):
        with st.spinner("جاري استخراج الإجابة..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
            db = FAISS.load_local("legal_vector_db_gemini", embeddings, allow_dangerous_deserialization=True)
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0)
            
            template = """أنت مستشار قانوني. أجب بناءً على النصوص التالية فقط:
            {context}
            السؤال: {question}
            الإجابة:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            rag_chain = (
                {"context": db.as_retriever() | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
                 "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )

            st.markdown("### 📝 الإجابة:")
            st.info(rag_chain.invoke(query))
    else:
        st.warning("يرجى الفهرسة أولاً.")
