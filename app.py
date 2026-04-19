import streamlit as st
import os
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
# يقرأ من Secrets في Streamlit Cloud أو من مدخل جانبي
openai_api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("يرجى إضافة مفتاح OpenAI API في الإعدادات للبدء.")
    st.stop()

# --- وظيفة معالجة ملفات PDF ---
def load_legal_docs(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
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

# --- القائمة الجانبية للفهرسة ---
lib_path = "laws_library"

with st.sidebar:
    st.header("⚙️ الإعدادات")
    if st.button("🚀 تحديث وفهرسة القوانين"):
        with st.spinner("جاري معالجة القوانين..."):
            raw_docs = load_legal_docs(lib_path)
            if raw_docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(raw_docs)
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local("legal_vector_db")
                st.success(f"تمت فهرسة {len(raw_docs)} صفحة قانونية!")
            else:
                st.error("المجلد 'laws_library' فارغ! أضف ملفات PDF أولاً.")

# --- محرك البحث الذكي ---
query = st.text_input("🔍 اسأل عن أي مقتضى قانوني:")

if query:
    if os.path.exists("legal_vector_db"):
        with st.spinner("جاري تحليل النصوص القانونية..."):
            # 1. تحميل قاعدة البيانات
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = FAISS.load_local("legal_vector_db", embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 5})

            # 2. إعداد القالب (Prompt)
            template = """أنت مستشار قانوني خبير. أجب على السؤال بناءً على النصوص القانونية التالية فقط:
            {context}
            
            السؤال: {question}
             الإجابة التفصيلية:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

            # 3. بناء السلسلة (الطريقة الحديثة LCEL)
            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # 4. التنفيذ والعرض
            response = rag_chain.invoke(query)
            docs = retriever.get_relevant_documents(query)
            
            st.markdown("### 📝 الإجابة القانونية:")
            st.info(response)
            
            st.markdown("---")
            st.subheader("📚 المصادر المستند إليها:")
            sources = set(f"🔹 {d.metadata['source']} (ص {d.metadata['page']})" for d in docs)
            for s in sources:
                st.write(s)
    else:
        st.warning("يرجى فهرسة القوانين من القائمة الجانبية أولاً.")
