import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# Load environment variables (API Key)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_PATH = "./data"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


class MyRAG: 
    def __init__(self, data_path=DATA_PATH):
        print("Initializing MyRAG system ...")
        self.data_path = data_path
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = Groq(api_key=GROQ_API_KEY)
        self.chunks = []
        self.metadata = []
        self.index = None
        self.bm25 = None
        self.tokenized_corpus = []
        
        # Build the knowledge base on startup
        self.load_data()

    # ================== 1. CHUNKING  ==================
    def chunk_text(self, text, max_chunk_size=1024):
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para.strip():
                continue

            para = re.sub(r'\s+', ' ', para)

            if len(para) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                for sent in sentences:
                    if len(temp_chunk) + len(sent) + 1 <= max_chunk_size:
                        if temp_chunk:
                            temp_chunk += " " + sent
                        else:
                            temp_chunk = sent
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sent
                if temp_chunk:
                    chunks.append(temp_chunk)
                continue

            separator_len = 2 if current_chunk else 0
            if len(current_chunk) + separator_len + len(para) <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # ================== 2. LOAD DATA & EMBED ==================
    def load_data(self):
        print("Loading and processing documents...")
        
        # Check if directory exists
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"Created directory {self.data_path}. Please add .md files here.")
            return

        for file in os.listdir(self.data_path):
            if file.endswith(".md"):
                with open(os.path.join(self.data_path, file), "r", encoding="utf-8") as f:
                    text = f.read()

                doc_chunks = self.chunk_text(text, max_chunk_size=1024)

                for chunk in doc_chunks:
                    self.chunks.append(chunk)
                    self.metadata.append({"source": file})

        if not self.chunks:
            print("No documents found to index.")
            return
        
        # --- BM25 ---
        print("Initializing BM25 Engine...")
        # Simple tokenization by splitting spaces
        self.tokenized_corpus = [chunk.split(" ") for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)


        # --- FAISS --- 
        print(f"Encoding {len(self.chunks)} chunks...")
        embeddings = self.model.encode(self.chunks, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"FAISS Index ready! Contains {self.index.ntotal} vectors.")



    # ================== 3. RETRIEVAL (HYBRID + RRF) ==================
    def retrieve(self, query, top_k=3, rrf_k=60):
        if not self.index or self.index.ntotal == 0:
            return []

        print(f"Running Hybrid Search (FAISS + BM25) for: {query}")
        
        # --- 1. FAISS (Dense) Search ---
        query_emb = self.model.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype(np.float32)
        
        # We pull top 10 to ensure good overlap for the fusion
        faiss_distances, faiss_indices = self.index.search(query_emb, 10) 
        
        faiss_ranks = {}
        for rank, idx in enumerate(faiss_indices[0]):
            if idx != -1:
                faiss_ranks[idx] = rank + 1  # Rank starts at 1, not 0

        # --- 2. BM25 (Sparse/Keyword) Search ---
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top 10 indices sorted by BM25 score descending
        bm25_indices = np.argsort(bm25_scores)[::-1][:10]
        
        bm25_ranks = {}
        for rank, idx in enumerate(bm25_indices):
            if bm25_scores[idx] > 0:  # Only rank it if BM25 actually found a keyword match
                bm25_ranks[idx] = rank + 1

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        # Get all unique document indices returned by both engines
        all_indices = set(faiss_ranks.keys()).union(set(bm25_ranks.keys()))
        
        for idx in all_indices:
            score = 0.0
            if idx in faiss_ranks:
                score += 1.0 / (rrf_k + faiss_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[idx])
            
            rrf_scores[idx] = score

        # Sort the documents based on their new RRF score (highest first)
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # --- 4. Format and Return Top K ---
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "score": rrf_scores[idx]
            })
            
        return results

    # ================== 4. ROUTING ==================
    def route_query_llm(self, query: str) -> str:
        system_prompt = """You are a highly accurate routing classifier for NileTel, an Egyptian telecom support system.
Analyze the user's query (which will be in Egyptian Arabic, English, or a mix) and classify it into EXACTLY ONE category.

Categories:
1. ticket: The user is complaining, angry, reporting an outage, or explicitly asking for technical support, an engineer, or a refund.
2. chat: The user is asking a general telecom question (internet speed, renewing bundles, balance, router setup) OR just saying hello/greetings.
3. out_of_scope: The user is asking about anything completely unrelated to telecom services like food (pizza), sports, movies, medical advice, etc.

Return ONLY the exact category name (ticket, chat, or out_of_scope). Do not add punctuation, explanations, or extra words."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant", # Upgraded to Groq's smartest model for accurate routing
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=0.0,
                max_tokens=10
            )

            # Extract raw content
            raw_content = response.choices[0].message.content
            
            # The repr() function shows hidden spaces, tabs, or newlines in the terminal
            print(f"RAW LLM RESPONSE: {repr(raw_content)}") 

            category = raw_content.lower()
            
            # Bulletproof extraction: Look for the word inside the string
            if "out_of_scope" in category:
                final_route = "out_of_scope"
            elif "ticket" in category:
                final_route = "ticket"
            else:
                final_route = "chat" # Default fallback
                
            print(f"Final LLM Route decided: {final_route}")
            return final_route

        except Exception as e:
            print(f"LLM Routing failed with error: {e}")
            return "chat"

    def route_query_hybrid(self, query: str) -> str:
        q = query.lower()
        words = q.split()

        TICKET_KEYWORDS = ["تذكر", "شكوى", "مهندس", "تصعيد", "فني", "تكت", "بلاغ", "اشتكي",
                           "صيان", "مش شغال", "مشكلة", "عطل", "مقطوع", "بطيء جدا", "خربان",
                           "ticket", "escalate", "complaint", "engineer"]

        GREETING_KEYWORDS = ["ازيك", "مرحبا", "hello", "hi", "شكرا", "سلام", "صباح", "مساء", 
                             "اهلا", "عامل ايه", "كيفك", "تحية"]
        
        TELECOM_KEYWORDS = ["نت", "انترنت", "internet", "dsl", "5g", "4g", "3g", "خط", "سرع", 
                            "فاتور", "رصيد", "باق", "شحن", "جدد", "راوتر", "مودم", "اتصالات", 
                            "niletel", "اشتراك", "خدم", "سيم", "بيانات", "data", "wifi", "واي فاي", "شبك"]
        
        # NEW SHIELD: Catch obvious garbage before it costs us an LLM call
        OUT_OF_SCOPE_KEYWORDS = ["بيتزا", "اكل", "مطعم", "دواء", "ماتش", "كورة", "اهلي", "زمالك", 
                                 "فيلم", "شاورما", "بيبسي", "طعام", "صداع", "علاج"]
        

        # 1. Fast Keyword Routing
        if any(kw in q for kw in OUT_OF_SCOPE_KEYWORDS):
            return "out_of_scope"
        if any(kw in q for kw in TICKET_KEYWORDS):
            return "ticket"
        
        if any(g in q for g in GREETING_KEYWORDS) and len(words) <= 4:
            return "chat"
        
        if any(term in q for term in TELECOM_KEYWORDS):
            return "chat"

        return self.route_query_llm(query)

    # ================== 5. GENERATE ANSWER ==================
    def generate_answer(self, query, retrieved_results):
        if not retrieved_results:
            return {
                "answer": "مش متأكد من البيانات المتاحة يا فندم.",
                "needs_action": "NO",
                "sources": [],
                "displayed_source": "Unknown"
            }

        context = "\n\n".join([f"Source: {res['source']}\n{res['text']}" for res in retrieved_results])
        
        prompt = f"""السياق المتاح (استخدمه فقط للإجابة):
{context}

سؤال العميل: {query}
"""
        

        system_ins = """أنت مساعد دعم عملاء محترف وودود في شركة NileTel للاتصالات.
قواعد صارمة جداً:
1. أجب باللهجة المصرية الطبيعية والمهنية (استخدم: يا فندم، تحت أمرك، هنساعدك).
2. استخدم فقط المعلومات الموجودة في "السياق المتاح". لا تخترع أي خطط، أسعار، أو أرقام من عندك.
3. الإجابة يجب أن تكون قصيرة ومباشرة في سطر واحد أو سطرين كحد أقصى.
4. يجب أن ينتهي ردك دائماً بالسطر: needs_action: YES أو needs_action: NO

متى تختار YES أو NO؟
- اختر needs_action: YES فقط إذا كان العميل يشتكي من عطل، يطلب مهندس، أو غاضب من الفاتورة ويريد تصعيد.
- اختر needs_action: NO للاستفسارات العادية (تجديد باقة، أسعار، سؤال عام).

مثال للإخراج المطلوب:
عرض رمضان بـ 3000 جنيه وبيديك 500 جيجا ومكالمات لا محدودة يا فندم، تحب أفعلهولك؟
needs_action: NO
"""


        print(f"Thinking ...")
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_ins}, {"role": "user", "content": prompt}],
            temperature=0.1
        )

        full_answer = response.choices[0].message.content.strip()

        # Robust extraction logic
        match = re.search(r"needs_action\s*[:=]\s*(YES|NO)", full_answer, re.IGNORECASE)
        needs_action = match.group(1).upper() if match else "NO"
        
        # Clean the answer text by removing the needs_action line completely
        clean_answer = re.sub(r"needs_action\s*[:=]\s*(YES|NO)", "", full_answer, flags=re.IGNORECASE).strip()

        best_source = retrieved_results[0]['source'] if retrieved_results else "Unknown"

        return {
            "answer": clean_answer,
            "needs_action": needs_action,
            "sources": list(set(r['source'] for r in retrieved_results)),
            "displayed_source": best_source
        }

    # ================== 6. MAIN PIPELINE ==================
    def run_rag_pipeline(self, query):
        route = self.route_query_hybrid(query)
        print(f"\n[{route.upper()} ROUTE] Query: {query}")
        
        if route == "out_of_scope":
            return {
                "answer": "أنا مختص فقط باستفسارات شركة NileTel للاتصالات يا فندم.", 
                "needs_action": "NO", 
                "sources": [],
                "displayed_source": "Unknown"
            }
            
        if route == "ticket":
            # Direct to ticket - skip retrieval to save time/tokens
            return {
                "answer": "تم استلام طلبك يا فندم، سيتم إنشاء تذكرة لمشكلتك والتواصل مع مهندس في أقرب وقت.", 
                "needs_action": "YES", 
                "sources": [],
                "displayed_source": "Ticket System"
            }

        # Normal chat flow
        results = self.retrieve(query, top_k=3)
        return self.generate_answer(query, results)