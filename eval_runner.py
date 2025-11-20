# eval_runner.py
import json
from tqdm import tqdm

from embedder import Embedder
from chunker import get_chunker
from vector_store import VectorStore
from llm import rewrite_query, answer_with_context
from router import classify_query
from multi_query import generate_multi_queries
from llm import client, DEFAULT_LLM_MODEL   # direct answer 용


############################################################
# 1. Document Loader
############################################################

def load_documents(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            docs.append(j)
    return docs


############################################################
# 2. Evaluation Query Loader
############################################################

def load_eval_queries(eval_path):
    queries = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            queries.append(j)
    return queries


############################################################
# 3. Direct Answer for GENERAL queries
############################################################

def llm_direct_answer(query, model=DEFAULT_LLM_MODEL):
    """
    Router에서 GENERAL로 분류된 경우,
    retrieval 없이 일반 답변을 생성하는 함수
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Provide a natural, conversational answer. "
                    "Do NOT use retrieval. Just answer normally."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.7
    )

    return resp.choices[0].message.content.strip()


############################################################
# 4. RAG Pipeline with Router + Multi Query
############################################################

def eval_rag(
        docs_path,
        eval_path,
        output_path="submission.jsonl",
        embed_model="BAAI/bge-m3",
        chunk_strategy="B",
        topk=5,
        count=5,   # 테스트 시는 5, 전체 실행 시 None
        multi_n=3  # multi-query 개수
):

    print("=== Loading Documents ===")
    documents = load_documents(docs_path)

    print("=== Initializing Embedder ===")
    embedder = Embedder(embed_model)

    print(f"=== Initializing Chunker: strategy {chunk_strategy} ===")
    chunk_fn = get_chunker(chunk_strategy)

    print("=== Building Vector Store ===")
    vs = VectorStore(embedder)
    vs.build(documents, chunk_fn=chunk_fn)

    print("=== Loading Evaluation Queries ===")
    queries = load_eval_queries(eval_path)

    print("=== Running Evaluation (RAG + Router + MultiQuery) ===")
    with open(output_path, "w", encoding="utf-8") as of:

        for i, item in enumerate(tqdm(queries, desc="RAG Evaluating")):

            # count 제한: smoke test
            if (count is not None) and (i >= count):
                print(f"[STOP] Count={count} reached. Ending early.")
                break

            messages = item["msg"]
            raw_query = " ".join(m["content"] for m in messages)

            # 1) Standalone query
            standalone = rewrite_query(messages)

            # 2) Routing
            query_type = classify_query(standalone)

            if query_type == "GENERAL":
                # 일반 대화: LLM direct answer, retrieval 없음
                topk_docids = []
                references = []
                answer = llm_direct_answer(standalone)

            else:
                # SCIENCE → Multi-query RAG

                # 2-1) Multi queries 생성
                multi_queries = generate_multi_queries(standalone, n=multi_n)

                # 원본 query도 포함 (안정성 ↑)
                all_queries = [standalone] + multi_queries

                # 2-2) 모든 쿼리로 retrieval
                candidate_idx = []
                for q in all_queries:
                    top_idx = vs.retrieve_topk(q, k=topk)
                    candidate_idx.extend(top_idx)

                # 중복 제거 + 순서 보존
                candidate_idx = list(dict.fromkeys(candidate_idx))

                # 상위 topk만 사용
                final_idx = candidate_idx[:topk]

                topk_docids = [vs.get_doc_id(i) for i in final_idx]
                references = [vs.get_chunk(i) for i in final_idx]

                # 2-3) Context 기반 LLM 답변
                answer = answer_with_context(standalone, references)

            # 3) Output serialization
            output = {
                "eval_id": item["eval_id"],
                "standalone_query": standalone,
                "topk": topk_docids,
                "answer": answer,
                "references": references
            }

            of.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"\n=== Done. Output saved to: {output_path} ===")



############################################################
# 5. Main Entry (Smoke Test)
############################################################

if __name__ == "__main__":

    # smoke test
    # eval_rag(
    #     docs_path="data/documents.jsonl",
    #     eval_path="data/eval.jsonl",
    #     output_path="submission_test.jsonl",
    #     embed_model="BAAI/bge-m3",
    #     chunk_strategy="B",
    #     topk=5,
    #     count=5,       # 먼저 5개만 테스트
    #     multi_n=3      # multi-query 3개 생성
    # )

    # 실제 전체 실행 시:
    eval_rag(
        docs_path="data/documents.jsonl",
        eval_path="data/eval.jsonl",
        output_path="submission_full.jsonl",
        embed_model="BAAI/bge-m3",
        chunk_strategy="B",
        topk=5,
        count=None,
        multi_n=3
    )
