import gradio as gr
import random, json, os
from dotenv import load_dotenv
from typing import List, TypedDict

# LangChain 및 LangGraph 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

# --- 기본 설정 (기존과 동일) ---
# 퀴즈 파일 및 출제 문항 개수 지정
QUIZ_FILE = "data/conan_quiz.json"
QUIZ_COUNT = 3

load_dotenv()
# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# --- LangGraph 채점 에이전트 정의 ---


# 1. 채점 결과의 데이터 구조를 Pydantic으로 정의
class GradingResult(BaseModel):
    """단일 문제에 대한 채점 결과"""

    question: str = Field(description="채점 대상 문제")
    correct_answer: str = Field(description="문제의 정답")
    user_answer: str = Field(description="사용자가 제출한 답변")
    is_correct: bool = Field(description="정답 여부")
    explanation: str = Field(description="정답에 대한 친절한 해설")


class FinalReport(BaseModel):
    """모든 문제에 대한 최종 채점 보고서"""

    results: List[GradingResult] = Field(description="각 문제별 채점 결과 리스트")
    total_score: str = Field(description="'총점: X/Y' 형식의 최종 점수 요약")


# 2. LangGraph의 상태(State) 정의
class GradingState(TypedDict):
    grading_input: str  # 채점을 위해 LLM에 전달될 전체 텍스트
    final_report: FinalReport  # LLM이 생성한 구조화된 채점 결과


# 3. 채점 로직을 수행할 노드(Node) 함수 정의
def grade_quiz(state: GradingState):
    """
    입력된 퀴즈 데이터에 대해 LLM을 호출하여 채점하고 상태를 업데이트합니다.
    """
    # Pydantic 모델을 기반으로 출력 파서 생성
    parser = PydanticOutputParser(pydantic_object=FinalReport)

    # LLM에게 전달할 프롬프트 템플릿 정의
    system_message = "당신은 '명탐정 코난' 퀴즈의 전문 채점관입니다. 주어진 문제, 정답, 사용자 답변을 바탕으로 채점해주세요. 각 문제에 대해 정답 여부를 판단하고 친절한 해설을 덧붙여주세요. 모든 채점이 끝나면, 마지막에는 '총점: X/Y' 형식으로 최종 점수를 반드시 요약해서 보여줘야 합니다. 반드시 지정된 JSON 형식으로만 답변해야 합니다."
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(
                "{grading_data}\n\n{format_instructions}"
            ),
        ],
        input_variables=["grading_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 프롬프트, LLM, 출력 파서를 연결하여 실행 가능한 체인(Runnable) 생성
    chain = prompt | llm | parser

    # 체인 실행
    report = chain.invoke({"grading_data": state["grading_input"]})

    # 계산된 결과를 상태에 업데이트하여 반환
    return {"final_report": report}


# 4. 그래프(Graph) 생성 및 컴파일
workflow = StateGraph(GradingState)
workflow.add_node("grader", grade_quiz)  # 노드 추가
workflow.set_entry_point("grader")  # 시작점 설정
workflow.add_edge("grader", END)  # 노드 실행 후 종료
grading_app = workflow.compile()  # 실행 가능한 앱으로 컴파일


# --- 퀴즈 로직 및 Gradio UI (대부분 기존과 동일) ---


# 퀴즈 로딩 함수
def load_quiz():
    with open(QUIZ_FILE, "r", encoding="utf-8") as f:
        all_q = json.load(f)
    return random.sample(all_q, QUIZ_COUNT)


# 문제 출력 - 선다형 구성
def get_question(state):
    idx = state["quiz_index"]
    q = state["questions"][idx]
    text = f"문제 {idx+1}: {q['question']}"
    if q["type"] == "multiple_choice":
        choices = [f"{i+1}. {c}" for i, c in enumerate(q["choices"])]
        text += "\n" + "\n".join(choices)
    return text


# 사용자 답변을 상태에 저장
def update_state(state, user_input):
    idx = state["quiz_index"]
    q = state["questions"][idx]
    processed = user_input.strip()

    if q["type"] == "multiple_choice":
        try:
            sel = int(processed) - 1
            if 0 <= sel < len(q["choices"]):
                processed = q["choices"][sel]
        except:
            pass

    state["user_answers"].append(
        {
            "question_text": q["question"],
            "user_response": processed,
            "is_correct": False,
            "correct_answer": str(q["answer"]),
        }
    )
    state["quiz_index"] += 1
    return state


# LangGraph Agent에게 전달할 채점 데이터 생성
def build_grading_input(state):
    parts = [
        "자, 이제 아래의 문제와 정답, 그리고 사용자의 답변을 보고 채점을 시작해주세요."
    ]
    for i, (q, a) in enumerate(zip(state["questions"], state["user_answers"])):
        parts.append(f"\n--- 문제 {i+1} ---")
        parts.append(f"문제: {q['question']}")
        if q["type"] == "multiple_choice":
            parts.append(f"선택지: {', '.join(q['choices'])}")
        parts.append(f"정답: {q['answer']}")
        parts.append(f"사용자 답변: {a['user_response']}")
    return "\n".join(parts)


# 퀴즈 시작 요청 처리
def handle_quiz_start(user_input, quiz_state, messages):
    quiz_state["questions"] = load_quiz()
    quiz_state["quiz_index"] = 0
    quiz_state["user_answers"] = []
    qtext = get_question(quiz_state)

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": qtext})

    return quiz_state, messages


# 퀴즈가 이미 끝난 경우
def handle_quiz_already_done(user_input, messages):
    bot_message = (
        "퀴즈가 이미 종료되었습니다. 다시 시작하려면 '퀴즈 시작'이라고 입력하세요."
    )
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": bot_message})
    return messages


# 사용자 답변 처리
def handle_user_answer(user_input, quiz_state, messages):
    quiz_state = update_state(quiz_state, user_input)
    messages.append({"role": "user", "content": user_input})

    if quiz_state["quiz_index"] < len(quiz_state["questions"]):
        # 다음 문제가 남은 경우
        qtext = get_question(quiz_state)
        messages.append({"role": "assistant", "content": qtext})
    else:
        # 모든 문제를 푼 경우, 채점 Agent 호출
        grading_input_data = build_grading_input(quiz_state)

        # [수정된 부분] LangGraph Agent 호출
        # 입력으로 상태 딕셔너리를 전달
        result_state = grading_app.invoke({"grading_input": grading_input_data})
        # 결과는 상태 딕셔너리의 'final_report' 키에 저장됨 (Pydantic 객체)
        final_report_obj = result_state["final_report"]

        # Pydantic 객체를 사용하여 사용자가 보기 좋은 형태의 문자열로 변환
        report_parts = ["채점이 완료되었습니다! 📝\n"]
        for i, res in enumerate(final_report_obj.results):
            is_correct_text = "정답" if res.is_correct else "오답"
            report_parts.append(f"--- 문제 {i+1} ---")
            report_parts.append(f"문제: {res.question}")
            report_parts.append(f"정답: {res.correct_answer}")
            report_parts.append(f"제출한 답변: {res.user_answer}")
            report_parts.append(f"결과: {is_correct_text}")
            report_parts.append(f"해설: {res.explanation}\n")

        report_parts.append(f"**{final_report_obj.total_score}**")
        final_report_str = "\n".join(report_parts)

        messages.append({"role": "assistant", "content": final_report_str})

    return quiz_state, messages


# 메인 챗봇 처리 함수
def chat_fn(user_input, state):
    user_input_lower = user_input.strip().lower()
    messages = state["chat_history"]
    quiz_state = state["quiz_state"]

    if not quiz_state["questions"]:
        if user_input_lower in ["퀴즈", "퀴즈 시작"]:
            quiz_state, messages = handle_quiz_start(user_input, quiz_state, messages)
        else:
            bot_message = "'퀴즈' 또는 '퀴즈 시작'이라고 입력하면 퀴즈를 시작합니다."
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": bot_message})

    elif quiz_state["quiz_index"] >= len(quiz_state["questions"]):
        messages = handle_quiz_already_done(user_input, messages)
    else:
        quiz_state, messages = handle_user_answer(user_input, quiz_state, messages)

    state["quiz_state"] = quiz_state
    state["chat_history"] = messages
    return messages, state


# 상태 초기화
def init_state():
    return {
        "quiz_state": {"quiz_index": 0, "questions": [], "user_answers": []},
        "chat_history": [],
    }


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("### 🕵️ 명탐정 코난 매니아 판별기 (LangGraph ver.)")
    chatbot = gr.Chatbot(
        label="명탐정 코난 퀴즈 챗봇",
        height=400,
        avatar_images=("data/avatar_user.png", "data/avatar_conan.png"),
        render=False,  # type 대신 render=False 사용
    )
    # render=False와 함께 Chatbot.like 사용을 위해 아래와 같이 수정
    chatbot.likeable = True
    chatbot.render()

    txt = gr.Textbox(placeholder="'퀴즈 시작'을 입력해보세요!", show_label=False)
    state = gr.State(init_state())

    txt.submit(chat_fn, inputs=[txt, state], outputs=[chatbot, state])
    txt.submit(lambda: "", None, txt)

demo.launch()
