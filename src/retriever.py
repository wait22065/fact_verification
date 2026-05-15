# ====== src/retriever.py ======
import wikipedia
import os
import re
import nltk
import torch
from sentence_transformers import SentenceTransformer, util

# ⚠️ 注意：如果是部署在云服务器上，需要注释掉这些本地代理配置
# 如果服务器能直连外网（如海外服务器），直接忽略
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

wikipedia.set_lang("en")
wikipedia.set_user_agent("FeverFactChecker/1.0 (Student_Project)")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True) # 加上 quiet 避免刷屏

_ST_MODEL = None

def get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        # 为了避免每次请求加载，保留单例。建议服务器启动时提前下载此模型
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _ST_MODEL

def retrieve_evidence_pipeline(claim, entity_query, top_k_sentences=3):
    try:
        search_results = wikipedia.search(entity_query, results=3)
        if not search_results:
            return f"ERROR: 找不到关键词 '{entity_query}' 相关的页面"

        all_sentences = []
        for title in search_results:
            try:
                content = wikipedia.page(title, auto_suggest=False).content
                sentences = nltk.sent_tokenize(content)
                clean_sentences = [s.strip() for s in sentences if len(s) > 20][:30]
                all_sentences.extend(clean_sentences)
            except Exception:
                continue

        if not all_sentences:
            return "ERROR: 无法从页面提取有效内容。"

        model = get_st_model()
        claim_embedding = model.encode(claim, convert_to_tensor=True)
        corpus_embeddings = model.encode(all_sentences, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(claim_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k_sentences, len(all_sentences)))
        
        evidence_chain = [all_sentences[idx] for idx in top_results[1]]
        return "\n".join([f"• {s}" for s in evidence_chain])

    except Exception as e:
        return f"RETRIEVAL_ERROR: {str(e)}"
        
def retrieve_evidence(entity_query, top_k=1, num_sentences=3):
    try:
        if re.search(r'[\u4e00-\u9fa5]', entity_query):
            wikipedia.set_lang("zh")
        else:
            wikipedia.set_lang("en")
        
        search_results = wikipedia.search(entity_query, results=top_k + 2)
        if not search_results:
            return "未找到相关证据。"

        all_evidence = []
        count = 0
        for title in search_results:
            if count >= top_k: break
            if any(x in title.lower() for x in ["list of", "列表", "album"]): continue
            try:
                summary = wikipedia.summary(title, sentences=num_sentences, auto_suggest=False)
                all_evidence.append(f"[来源 {count+1}: {title}]\n{summary}")
                count += 1
            except:
                continue
        
        return "\n\n".join(all_evidence) if all_evidence else "未找到有效证据。"
    except Exception as e:
        return f"检索错误: {str(e)}"