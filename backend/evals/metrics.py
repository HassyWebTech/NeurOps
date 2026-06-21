import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rag.retrieve import retrieve


# ── Test Suite ──
# Each test case defines:
# - question: what we ask the agent
# - expected_tool: which tool the agent should call
# - expected_keywords: words we expect in the answer
# - relevant_keywords: words we expect in retrieved reviews

TEST_CASES = [
    {
        "id": "TC001",
        "question": "What are customers saying about late deliveries?",
        "expected_tool": "search_customer_reviews",
        "expected_keywords": ["delivery", "late", "order"],
        "relevant_keywords": ["entrega", "delivery", "prazo", "atraso"]
    },
    {
        "id": "TC002",
        "question": "Which customers are most at risk of churning?",
        "expected_tool": "churn_analysis",
        "expected_keywords": ["churn", "risk", "days", "purchase"],
        "relevant_keywords": ["churn", "risk", "customer"]
    },
    {
        "id": "TC003",
        "question": "What are my top revenue generating product categories?",
        "expected_tool": "customer_analytics",
        "expected_keywords": ["revenue", "category", "product"],
        "relevant_keywords": ["revenue", "category", "sales"]
    },
    {
        "id": "TC004",
        "question": "Generate a re-engagement campaign for churning customers",
        "expected_tool": "generate_campaign",
        "expected_keywords": ["campaign", "message", "offer"],
        "relevant_keywords": ["campaign", "engagement", "customers"]
    },
    {
        "id": "TC005",
        "question": "What do customers love most about products?",
        "expected_tool": "search_customer_reviews",
        "expected_keywords": ["product", "customers", "love"],
        "relevant_keywords": ["produto", "bom", "otimo", "excelente"]
    }
]


def evaluate_retrieval(test_cases: list) -> dict:
    """
    Evaluates RAG retrieval quality.

    For each test case:
    1. Retrieves top-5 reviews using the question
    2. Checks if retrieved reviews contain relevant keywords
    3. Measures retrieval latency
    4. Returns accuracy score and per-test details
    """
    results = []
    total_score = 0

    for tc in test_cases:
        if tc["expected_tool"] != "search_customer_reviews":
            continue

        start = time.time()
        retrieved = retrieve(query=tc["question"], top_k=5)
        latency = round(time.time() - start, 3)

        # Check if any retrieved review contains relevant keywords
        hits = 0
        for result in retrieved:
            text_lower = result["text"].lower()
            if any(kw in text_lower for kw in tc["relevant_keywords"]):
                hits += 1

        # Score: ratio of relevant results retrieved
        score = hits / len(retrieved) if retrieved else 0
        total_score += score

        results.append({
            "test_id": tc["id"],
            "question": tc["question"],
            "retrieved_count": len(retrieved),
            "relevant_hits": hits,
            "score": round(score, 2),
            "latency_seconds": latency,
            "avg_similarity": round(
                sum(r["similarity"] for r in retrieved) / len(retrieved), 4
            ) if retrieved else 0
        })

    retrieval_tests = [tc for tc in test_cases
                       if tc["expected_tool"] == "search_customer_reviews"]
    avg_score = total_score / len(retrieval_tests) if retrieval_tests else 0

    return {
        "retrieval_accuracy": round(avg_score * 100, 1),
        "tests_run": len(results),
        "results": results
    }


def evaluate_tool_selection(agent_results: list) -> dict:
    """
    Evaluates whether the agent selected the correct tool
    for each test case.

    agent_results is a list of dicts:
    [{"test_id": "TC001", "tool_called": "search_customer_reviews"}, ...]
    """
    if not agent_results:
        return {
            "tool_accuracy": 0,
            "tests_run": 0,
            "results": []
        }

    correct = 0
    results = []

    tc_map = {tc["id"]: tc for tc in TEST_CASES}

    for ar in agent_results:
        tc = tc_map.get(ar["test_id"])
        if not tc:
            continue

        is_correct = ar["tool_called"] == tc["expected_tool"]
        if is_correct:
            correct += 1

        results.append({
            "test_id": ar["test_id"],
            "question": tc["question"],
            "expected_tool": tc["expected_tool"],
            "tool_called": ar["tool_called"],
            "correct": is_correct
        })

    accuracy = correct / len(results) if results else 0

    return {
        "tool_accuracy": round(accuracy * 100, 1),
        "tests_run": len(results),
        "correct": correct,
        "results": results
    }


def evaluate_response_quality(question: str, answer: str,
                               expected_keywords: list) -> dict:
    """
    Evaluates response quality by checking if the answer
    contains expected keywords — a simple proxy for
    groundedness and relevance.

    In production, this would use an LLM-as-judge approach
    (a second LLM grades the first LLM's response).
    We use keyword matching here to avoid extra API calls.
    """
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw in answer_lower)
    score = hits / len(expected_keywords) if expected_keywords else 0

    return {
        "question": question,
        "keywords_expected": expected_keywords,
        "keywords_found": [kw for kw in expected_keywords
                           if kw in answer_lower],
        "quality_score": round(score * 100, 1)
    }


def run_retrieval_evals() -> dict:
    """
    Runs the full retrieval evaluation suite.
    Called by the /api/evals endpoint.
    """
    print("Running retrieval evaluations...")
    retrieval_results = evaluate_retrieval(TEST_CASES)

    return {
        "eval_type": "retrieval",
        "overall_score": retrieval_results["retrieval_accuracy"],
        "summary": {
            "retrieval_accuracy": f"{retrieval_results['retrieval_accuracy']}%",
            "tests_run": retrieval_results["tests_run"],
        },
        "details": retrieval_results["results"],
        "test_cases_total": len(TEST_CASES),
        "notes": {
            "retrieval": "Measures if retrieved reviews contain keywords relevant to the query",
            "tool_selection": "Requires agent run — call /api/evals/agent for full suite",
            "response_quality": "Keyword-based proxy. Production: LLM-as-judge"
        }
    }