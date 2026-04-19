import streamlit as st
import os
from PyPDF2 import PdfReader

# --- استدعاءات LangChain الحديثة ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المستشار القانوني للجماعات", layout="wide")
st.title("⚖️ منصة الذكاء الاصطناعي للقوانين الجماعية")

# --- إعدادات الأمان ومفاتيح API ---
# سيحاول التطبيق القراءة من Secrets أولاً، وإذا لم يجدها سيطلبها من المستخدم
openai_api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("يرجى إضافة مفتاح OpenAI API للبدء.")
    st.stop()

# --- وظيفة معالجة الملفات ---
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
            st.error(f"خطأ في قراءة الملف {filename}: {e}")
    return documents

# --- القائمة الجانبية ---
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
                st.success(f"تمت فهرسة {len(raw_docs)} صفحة بنجاح!")
            else:
                st.error("المجلد فارغ! يرجى إضافة ملفات PDF في مجلد laws_library")

# --- محرك البحث والاسترجاع ---
query = st.text_input("🔍 اسأل عن أي مقتضى قانوني:")

if query:
    if os.path.exists("legal_vector_db"):
        with st.spinner("جاري البحث في النصوص القانونية..."):
            # 1. تحميل قاعدة البيانات
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = FAISS.load_local("legal_vector_db", embeddings, allow_dangerous_deserialization=True)
            
            # 2. إعداد القالب (Prompt)
            system_prompt = (
                "أنت مستشار قانوني خبير. استخدم النصوص القانونية المقدمة فقط للإجابة على السؤال. "
                "إذا لم تجد الإجابة في النصوص، قل أنك لا تعرف، لا تحاول اختراع إجابة."
                "\n\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # 3. بناء سلسلة الاسترجاع (Retrieval Chain)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 5}), combine_docs_chain)
            
            # 4. تنفيذ البحث
            response = retrieval_chain.invoke({"input": query})
            
            # 5. عرض النتائج
            st.markdown("### 📝 الإجابة القانونية:")
            st.info(response["answer"])
            
            st.markdown("---")
            st.subheader("📚 المصادر المعتمدة:")
            # عرض المصادر بدون تكرار
            sources = set()
            for doc in response["context"]:
                sources.add(f"🔹 {doc.metadata['source']} (ص {doc.metadata['page']})")
            for source in sources:
                st.write(source)
    else:
        st.warning("يرجى الضغط على زر 'تحديث وفهرسة القوانين' أولاً.")
