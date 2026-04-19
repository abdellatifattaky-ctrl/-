import streamlit as st
import os
import zipfile
from PyPDF2 import PdfReader

# استدعاءات LangChain الحديثة والمستقرة
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المستشار القانوني للجماعات", layout="wide")
st.title("⚖️ منصة الذكاء الاصطناعي للقوانين الجماعية")

# --- إعدادات مفتاح API ---
openai_api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("يرجى إضافة مفتاح OpenAI API في الإعدادات للبدء.")
    st.stop()

# --- وظيفة فك الضغط تلقائياً ---
def handle_zip_file(zip_path, extract_to="laws_library"):
    if os.path.exists(zip_path):
        with st.spinner("جاري فك ضغط ملف القوانين..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        return True
    return False

# --- وظيفة معالجة ملفات PDF من مجلد ---
def load_legal_docs(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        return None
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not files:
        return None

    for filename in files:
        path = os.path.join(folder_path, filename)
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
        except Exception as e:
            st.error(f"خطأ في قراءة {filename}: {e}")
    return documents

# --- القائمة الجانبية ---
lib_path = "laws_library"
zip_file_name = "laws_library.zip"

with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # محاولة فك الضغط إذا وجد الملف
    if os.path.exists(zip_file_name):
        handle_zip_file(zip_file_name, lib_path)
        st.success("✅ تم العثور على ملف ZIP وفكه بنجاح.")

    # خيار إضافي لرفع ملفات يدوياً
    uploaded_files = st.file_uploader("أو ارفع ملفات PDF جديدة:", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 بدء الفهرسة وتحديث البيانات"):
        all_docs = []
        
        # 1. معالجة الملفات المرفوعة يدوياً
        if uploaded_files:
            for f in uploaded_files:
                reader = PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        all_docs.append(Document(page_content=text, metadata={"source": f.name, "page": i + 1}))
        
        # 2. معالجة الملفات الموجودة في المجلد (الناتجة عن ZIP أو المرفوعة مسبقاً)
        folder_docs = load_legal_docs(lib_path)
        if folder_docs:
            all_docs.extend(folder_docs)

        if all_docs:
            with st.spinner("جاري إنشاء قاعدة البيانات الذكية..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(all_docs)
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local("legal_vector_db")
                st.success(f"تمت فهرسة {len(all_docs)} صفحة قانونية بنجاح!")
        else:
            st.error("لم يتم العثور على أي ملفات PDF (لا في المجلد ولا في الرفع اليدوي)!")

# --- محرك البحث ---
query = st.text_input("🔍 اسأل عن أي مقتضى قانوني (مثلاً: اختصاصات رئيس الجماعة):")

if query:
    if os.path.exists("legal_vector_db"):
        with st.spinner("جاري تحليل النصوص..."):
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = FAISS.load_local("legal_vector_db", embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 5})

            template = """أنت مستشار قانوني خبير في القوانين التنظيمية للجماعات. 
            أجب بدقة بناءً على النصوص التالية فقط:
            {context}
            السؤال: {question}
            الإجابة:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

            rag_chain = (
                {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
                 "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )

            response = rag_chain.invoke(query)
            st.markdown("### 📝 الإجابة القانونية:")
            st.info(response)
            
            # عرض المصادر
            with st.expander("📚 عرض المصادر والمراجع"):
                docs = retriever.get_relevant_documents(query)
                for d in docs:
                    st.write(f"📍 **{d.metadata['source']}** - صفحة {d.metadata['page']}")
    else:
        st.warning("⚠️ قاعدة البيانات غير جاهزة. يرجى الضغط على 'بدء الفهرسة' من القائمة الجانبية.")
