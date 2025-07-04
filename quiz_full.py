import gradio as gr
import random
import json
import os
from dotenv import load_dotenv
from typing import List, TypedDict

# LangChain 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

# --- 초기 설정 및 데이터 로딩 ---

QUIZ_FILE = "data/conan_quiz.json"
QUIZ_COUNT = 3

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# 퀴즈 로딩 함수
def load_quiz():
    with open(QUIZ_FILE, "r", encoding="utf-8") as f:
        all_q = json.load(f)
    return random.sample(all_q, QUIZ_COUNT)


# --- Pydantic 모델 및 State 정의 ---


class GradingResult(BaseModel):
    question: str = Field(description="채점 대상 문제")
    correct_answer: str = Field(description="문제의 정답")
    user_answer: str = Field(description="사용자가 제출한 답변")
    is_correct: bool = Field(description="정답 여부")
    explanation: str = Field(description="정답에 대한 친절한 해설")


class FinalReport(BaseModel):
    results: List[GradingResult] = Field(description="각 문제별 채점 결과 리스트")
    total_score: str = Field(description="'총점: X/Y' 형식의 최종 점수 요약")


class QuizState(TypedDict):
    user_input: str
    questions: List[dict]
    user_answers: List[str]
    quiz_index: int
    chat_history: List[tuple]
    grading_input_str: str | None
    final_report: FinalReport | None


# --- StateGraph 노드 함수 정의 ---


def start_quiz(state: QuizState) -> QuizState:
    """퀴즈를 시작하고 상태를 초기화합니다."""
    questions = load_quiz()
    if not questions:
        state["chat_history"].append(
            ("assistant", "퀴즈를 불러오는 데 실패했거나 풀 수 있는 문제가 없습니다.")
        )
        state["questions"] = []
        return state

    state["questions"] = questions
    state["quiz_index"] = 0
    state["user_answers"] = []
    state["final_report"] = None
    state["chat_history"].append(("assistant", "명탐정 코난 퀴즈를 시작합니다! 🕵️‍♂️"))
    return state


def ask_question(state: QuizState) -> QuizState:
    """현재 quiz_index에 맞는 문제를 포맷하여 chat_history에 추가합니다."""
    idx = state["quiz_index"]
    q = state["questions"][idx]

    text = f"문제 {idx + 1}: {q['question']}"
    if q["type"] == "multiple_choice":
        choices = [f"{i + 1}. {c}" for i, c in enumerate(q["choices"])]
        text += "\n" + "\n".join(choices)

    state["chat_history"].append(("assistant", text))
    return state


def process_and_store_answer(state: QuizState) -> QuizState:
    """사용자 답변을 처리하고 저장한 뒤, 다음 문제로 넘어갑니다."""
    idx = state["quiz_index"]
    q = state["questions"][idx]
    user_input = state["user_input"].strip()

    processed_answer = user_input
    if q["type"] == "multiple_choice":
        try:
            sel = int(user_input) - 1
            if 0 <= sel < len(q["choices"]):
                processed_answer = q["choices"][sel]
        except (ValueError, IndexError):
            pass

    state["user_answers"].append(processed_answer)
    state["quiz_index"] += 1
    return state


def prepare_grading_prompt(state: QuizState) -> QuizState:
    """채점을 위해 LLM에 전달할 프롬프트를 생성합니다."""
    state["chat_history"].append(
        (
            "assistant",
            "모든 문제를 다 푸셨군요! 잠시만 기다리시면 채점해 드릴게요... 📝",
        )
    )
    parts = [
        "자, 이제 아래의 문제와 정답, 그리고 사용자의 답변을 보고 채점을 시작해주세요."
    ]
    for i, (q, a) in enumerate(zip(state["questions"], state["user_answers"])):
        parts.append(f"\n--- 문제 {i + 1} ---")
        parts.append(f"문제: {q['question']}")
        if q["type"] == "multiple_choice":
            parts.append(f"선택지: {', '.join(q['choices'])}")
        parts.append(f"정답: {q['answer']}")
        parts.append(f"사용자 답변: {a}")

    state["grading_input_str"] = "\n".join(parts)
    return state


def grade_with_llm_and_parse(state: QuizState) -> QuizState:
    """LLM을 호출하여 채점하고 결과를 파싱합니다."""
    parser = PydanticOutputParser(pydantic_object=FinalReport)
    system_message = "당신은 '명탐정 코난' 퀴즈의 전문 채점관입니다. 주어진 문제, 정답, 사용자 답변을 바탕으로 채점해주세요. 각 문제에 대해 정답 여부를 판단하고 친절한 해설을 덧붙여주세요. 모든 채점이 끝나면, 마지막에는 '총점: X/Y' 형식으로 최종 점수를 반드시 요약해서 보여줘야 합니다. 반드시 지정된 JSON 형식으로만 답변해야 합니다."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{grading_data}\n\n{format_instructions}"),
        ]
    )

    try:
        chain_input = {
            "grading_data": state["grading_input_str"],
            "format_instructions": parser.get_format_instructions(),
        }
        formatted_prompt = prompt.format_prompt(**chain_input)
        response = llm.invoke(formatted_prompt)
        report = parser.invoke(response)
        state["final_report"] = report
    except Exception as e:
        print(f"채점 중 오류 발생: {e}")
        error_report = FinalReport(results=[], total_score="채점 오류가 발생했습니다.")
        state["final_report"] = error_report

    return state


def format_final_report(state: QuizState) -> QuizState:
    """파싱된 최종 리포트를 사용자에게 보여줄 문자열로 변환합니다."""
    final_report_obj = state["final_report"]
    report_parts = ["채점이 완료되었습니다! 🎉\n"]

    if final_report_obj and final_report_obj.results:
        for i, res in enumerate(final_report_obj.results):
            is_correct_text = "✅ 정답" if res.is_correct else "❌ 오답"
            report_parts.append(f"--- 문제 {i + 1} ---")
            report_parts.append(f"문제: {res.question}")
            report_parts.append(f"정답: {res.correct_answer}")
            report_parts.append(f"제출한 답변: {res.user_answer}")
            report_parts.append(f"결과: {is_correct_text}")
            report_parts.append(f"해설: {res.explanation}\n")
        report_parts.append(f"**{final_report_obj.total_score}**")
    else:
        report_parts.append("채점 결과를 생성하는 데 실패했습니다.")

    report_parts.append("\n퀴즈를 다시 시작하려면 '퀴즈 시작'이라고 입력해주세요.")
    state["chat_history"].append(("assistant", "\n".join(report_parts)))
    return state


def handle_invalid_start(state: QuizState) -> QuizState:
    """퀴즈 시작 명령어가 아닐 경우 안내 메시지를 추가합니다."""
    bot_message = "'퀴즈' 또는 '퀴즈 시작'이라고 입력하면 퀴즈가 시작됩니다."
    state["chat_history"].append(("assistant", bot_message))
    return state


# --- StateGraph 조건부 함수 ---


def should_continue_quiz(state: QuizState) -> str:
    """퀴즈를 계속할지, 채점을 시작할지 결정합니다."""
    if state["quiz_index"] < len(state["questions"]):
        return "continue_quiz"
    else:
        return "grade_quiz"


def route_initial_input(state: QuizState) -> str:
    """사용자의 입력을 분석하여 워크플로우의 시작점을 결정합니다."""
    if state.get("questions") and state["questions"]:
        return "process_answer"
    else:
        if state["user_input"].strip().lower() in ["퀴즈", "퀴즈 시작"]:
            return "start_quiz"
        else:
            return "invalid_start"


# --- StateGraph 정의 및 컴파일 ---

workflow = StateGraph(QuizState)

# 노드 추가
workflow.add_node("start_quiz", start_quiz)
workflow.add_node("ask_question", ask_question)
workflow.add_node("process_answer", process_and_store_answer)
workflow.add_node("prepare_grading", prepare_grading_prompt)
workflow.add_node("grade_and_parse", grade_with_llm_and_parse)
workflow.add_node("format_report", format_final_report)
workflow.add_node("invalid_start", handle_invalid_start)

# === 오류 수정: 조건부 진입점 설정 ===
workflow.set_conditional_entry_point(
    route_initial_input,
    {
        "start_quiz": "start_quiz",
        "process_answer": "process_answer",
        "invalid_start": "invalid_start",
    },
)

# 엣지 연결
workflow.add_edge("start_quiz", "ask_question")
workflow.add_edge("ask_question", END)
workflow.add_edge("invalid_start", END)

workflow.add_conditional_edges(
    "process_answer",
    should_continue_quiz,
    {"continue_quiz": "ask_question", "grade_quiz": "prepare_grading"},
)
workflow.add_edge("prepare_grading", "grade_and_parse")
workflow.add_edge("grade_and_parse", "format_report")
workflow.add_edge("format_report", END)

quiz_app = workflow.compile()


# --- Gradio UI 및 인터페이스 함수 ---


def init_state():
    return {"quiz_state": {"questions": [], "chat_history": []}}


def chat_fn(user_input, state):
    quiz_state = state["quiz_state"]

    if quiz_state.get("final_report") and user_input.strip().lower() in [
        "퀴즈",
        "퀴즈 시작",
    ]:
        quiz_state = init_state()["quiz_state"]

    current_chat_history = quiz_state.get("chat_history", [])
    current_chat_history.append(("user", user_input))

    graph_input = {
        **quiz_state,
        "user_input": user_input,
        "chat_history": current_chat_history,
    }

    new_state = quiz_app.invoke(graph_input)

    state["quiz_state"] = new_state

    chat_display = [
        {"role": role, "content": content}
        for role, content in new_state["chat_history"]
    ]

    return chat_display, state


# --- UI 정의 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("### 🕵️ 명탐정 코난 매니아 판별기 (LangGraph ver.)")

    chatbot = gr.Chatbot(
        label="명탐정 코난 퀴즈 챗봇",
        height=500,
        avatar_images=("data/avatar_user.png", "data/avatar_conan.png"),
        type="messages",
    )

    txt = gr.Textbox(
        placeholder="'퀴즈 시작'을 입력해보세요!", show_label=False, container=False
    )
    state = gr.State(init_state())
    txt.submit(chat_fn, inputs=[txt, state], outputs=[chatbot, state])
    txt.submit(lambda: "", None, txt, queue=False)

demo.launch(debug=True)
