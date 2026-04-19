import streamlit as st
import os
import zipfile
import time
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
st.set_page_config(page_title="المستشار القانوني (Gemini Edition)", layout="wide")
st.title("⚖️ منصة الذكاء الاصطناعي للقوانين الجماعية")
st.markdown("---")

# --- إعدادات مفتاح API ---
gemini_api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not gemini_api_key:
    st.warning("⚠️ يرجى إضافة مفتاح Gemini API في الإعدادات (Secrets) للبدء.")
    st.stop()

# --- وظائف المساعدة ---
def handle_zip_file(zip_path, extract_to="laws_library"):
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except: return False
    return False

def load_docs_from_folder(folder_path):
    documents = []
    if not os.path.exists(folder_path): return []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            path = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
            except: continue
    return documents

# --- القائمة الجانبية (الفهرسة) ---
with st.sidebar:
    st.header("⚙️ إدارة البيانات")
    
    # فك الضغط تلقائياً إذا وجد الملف
    if os.path.exists("laws_library.zip"):
        handle_zip_file("laws_library.zip")
    
    uploaded_files = st.file_uploader("رفع ملفات قانونية جديدة (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 بدء الفهرسة الذكية"):
        all_docs = []
        # تجميع الملفات المرفوعة
        if uploaded_files:
            for f in uploaded_files:
                reader = PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text: all_docs.append(Document(page_content=text, metadata={"source": f.name, "page": i + 1}))
        
        # تجميع الملفات من المجلد
        all_docs.extend(load_docs_from_folder("laws_library"))

        if all_docs:
            with st.spinner("جاري معالجة النصوص وتجنب الحظر..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                chunks = splitter.split_documents(all_docs)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
                
                # الفهرسة على دفعات لتجنب RateLimitError
                batch_size = 30 
                vector_db = None
                progress_bar = st.progress(0)
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    if vector_db is None:
                        vector_db = FAISS.from_documents(batch, embeddings)
                    else:
                        vector_db.add_documents(batch)
                    
                    # تحديث التقدم وانتظار بسيط لتفادي ضغط الـ API
                    progress_bar.progress(min((i + batch_size) / len(chunks), 1.0))
                    time.sleep(1.5) 
                
                vector_db.save_local("legal_vector_db_gemini")
                st.success("✅ تمت الفهرسة بنجاح! يمكنك الآن طرح الأسئلة.")
        else:
            st.error("لم يتم العثور على ملفات PDF للمعالجة.")

# --- محرك البحث والاسترجاع ---
query = st.text_input("🔍 اسأل عن أي مقتضى قانوني (مثلاً: ما هي اختصاصات رئيس المجلس؟)")

if query:
    if os.path.exists("legal_vector_db_gemini"):
        with st.spinner("جاري استنباط الإجابة القانونية..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
            db = FAISS.load_local("legal_vector_db_gemini", embeddings, allow_dangerous_deserialization=True)
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0)
            
            prompt = ChatPromptTemplate.from_template("""
            أنت مستشار قانوني خبير في القوانين التنظيمية.
            أجب على السؤال بناءً على المراجع التالية فقط. إذا لم تجد الإجابة، قل لا أعرف.
            
            المراجع:
            {context}
            
            السؤال: {question}
            الإجابة القانونية المفصلة:""")
            
            chain = (
                {"context": db.as_retriever(search_kwargs={"k": 5}) | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
                 "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            
            answer = chain.invoke(query)
            st.markdown("### 📝 النتيجة:")
            st.info(answer)
            
            # عرض المصادر
            with st.expander("📚 المراجع المستخدمة في الإجابة"):
                relevant_docs = db.similarity_search(query, k=5)
                for d in relevant_docs:
                    st.write(f"🔹 **{d.metadata['source']}** - صفحة {d.metadata['page']}")
    else:
        st.warning("⚠️ قاعدة البيانات غير جاهزة. يرجى فهرسة الملفات من القائمة الجانبية أولاً.")
