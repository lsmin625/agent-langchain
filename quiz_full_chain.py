import gradio as gr
import random, json, os
from dotenv import load_dotenv
from typing import List, TypedDict

# LangChain 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

QUIZ_FILE = "data/conan_quiz.json"
QUIZ_COUNT = 3

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def load_quiz():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(QUIZ_FILE):
        default_quiz = [
            {
                "type": "short_answer",
                "question": "코난의 본명은 무엇일까요?",
                "answer": "쿠도 신이치",
            },
            {
                "type": "multiple_choice",
                "question": "코난을 작아지게 만든 약의 이름은?",
                "choices": ["APTX4869", "APTX4868", "APTX4870"],
                "answer": "APTX4869",
            },
            {
                "type": "short_answer",
                "question": "란의 아버지는 직업은 무엇일까요?",
                "answer": "탐정",
            },
        ]
        with open(QUIZ_FILE, "w", encoding="utf-8") as f:
            json.dump(default_quiz, f, ensure_ascii=False, indent=2)
    with open(QUIZ_FILE, "r", encoding="utf-8") as f:
        all_q = json.load(f)
    count = min(QUIZ_COUNT, len(all_q))
    return random.sample(all_q, count)


class GradingResult(BaseModel):
    question: str = Field(description="채점 대상 문제")
    correct_answer: str = Field(description="문제의 정답")
    user_answer: str = Field(description="사용자가 제출한 답변")
    is_correct: bool = Field(description="정답 여부")
    explanation: str = Field(description="정답에 대한 친절한 해설")


class FinalReport(BaseModel):
    results: List[GradingResult] = Field(description="각 문제별 채점 결과 리스트")
    total_score: str = Field(description="'총점: X/Y' 형식의 최종 점수 요약")


class GradingState(TypedDict):
    grading_input: str
    final_report: FinalReport


def grade_quiz(state: GradingState):
    parser = PydanticOutputParser(pydantic_object=FinalReport)
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
    chain = prompt | llm | parser
    try:
        report = chain.invoke({"grading_data": state["grading_input"]})
        return {"final_report": report}
    except Exception as e:
        print(f"채점 중 오류 발생: {e}")
        error_report = FinalReport(results=[], total_score="채점 오류")
        return {"final_report": error_report}


workflow = StateGraph(GradingState)
workflow.add_node("grader", grade_quiz)
workflow.set_entry_point("grader")
workflow.add_edge("grader", END)
grading_app = workflow.compile()


def get_question(state):
    idx = state["quiz_index"]
    q = state["questions"][idx]
    text = f"문제 {idx+1}: {q['question']}"
    if q["type"] == "multiple_choice":
        choices = [f"{i+1}. {c}" for i, c in enumerate(q["choices"])]
        text += "\n" + "\n".join(choices)
    return text


def update_state(state, user_input):
    idx = state["quiz_index"]
    q = state["questions"][idx]
    processed = user_input.strip()
    if q["type"] == "multiple_choice":
        try:
            sel = int(processed) - 1
            if 0 <= sel < len(q["choices"]):
                processed = q["choices"][sel]
        except (ValueError, IndexError):
            pass
    state["user_answers"].append({"user_response": processed})
    state["quiz_index"] += 1
    return state


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


def handle_quiz_start(user_input, quiz_state, messages):
    quiz_state["questions"] = load_quiz()
    if not quiz_state["questions"]:
        messages.append(("user", user_input))
        messages.append(
            ("assistant", "퀴즈를 불러오는 데 실패했거나 풀 수 있는 문제가 없습니다.")
        )
        return quiz_state, messages
    quiz_state["quiz_index"] = 0
    quiz_state["user_answers"] = []
    qtext = get_question(quiz_state)
    messages.append(("user", user_input))
    messages.append(("assistant", qtext))
    return quiz_state, messages


def handle_quiz_already_done(user_input, messages):
    bot_message = "퀴즈가 이미 종료되었습니다. 퀴즈를 다시 시작하려면 '퀴즈 시작'이라고 입력해주세요."
    messages.append(("user", user_input))
    messages.append(("assistant", bot_message))
    return messages


def handle_user_answer(user_input, quiz_state, messages):
    quiz_state = update_state(quiz_state, user_input)
    messages.append(("user", user_input))
    if quiz_state["quiz_index"] < len(quiz_state["questions"]):
        qtext = get_question(quiz_state)
        messages.append(("assistant", qtext))
    else:
        grading_input_data = build_grading_input(quiz_state)
        result_state = grading_app.invoke({"grading_input": grading_input_data})
        final_report_obj = result_state["final_report"]
        report_parts = ["채점이 완료되었습니다! 📝\n"]
        if final_report_obj and final_report_obj.results:
            for i, res in enumerate(final_report_obj.results):
                is_correct_text = "정답" if res.is_correct else "오답"
                report_parts.append(f"--- 문제 {i+1} ---")
                report_parts.append(f"문제: {res.question}")
                report_parts.append(f"정답: {res.correct_answer}")
                report_parts.append(f"제출한 답변: {res.user_answer}")
                report_parts.append(f"결과: {is_correct_text}")
                report_parts.append(f"해설: {res.explanation}\n")
            report_parts.append(f"**{final_report_obj.total_score}**")
        else:
            report_parts.append("채점 결과를 생성하는 데 실패했습니다.")
        final_report_str = "\n".join(report_parts)
        messages.append(("assistant", final_report_str))
    return quiz_state, messages


def chat_fn(user_input, state):
    messages = state["chat_history"].copy()
    quiz_state = state["quiz_state"]
    user_input_lower = user_input.strip().lower()

    # 퀴즈 상태에 따른 분기 로직
    if not quiz_state.get("questions") or not quiz_state.get("questions"):
        if user_input_lower in ["퀴즈", "퀴즈 시작"]:
            quiz_state, messages = handle_quiz_start(user_input, quiz_state, messages)
        else:
            bot_message = "'퀴즈' 또는 '퀴즈 시작'이라고 입력하면 퀴즈를 시작합니다."
            messages.append(("user", user_input))
            messages.append(("assistant", bot_message))
    elif quiz_state["quiz_index"] >= len(quiz_state["questions"]):
        if user_input_lower in ["퀴즈", "퀴즈 시작"]:
            quiz_state, messages = handle_quiz_start(user_input, quiz_state, messages)
        else:
            messages = handle_quiz_already_done(user_input, messages)
    else:
        quiz_state, messages = handle_user_answer(user_input, quiz_state, messages)

    # state 업데이트
    state["quiz_state"] = quiz_state
    state["chat_history"] = messages

    # ### Gradio 챗봇의 'value' 형식 변환 ###
    # 내부 메시지 형식 [('user', ...), ('assistant', ...)]을
    # 새로운 표준 형식 [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]으로 변환합니다.
    chat_display = []
    for role, content in messages:
        chat_display.append({"role": role, "content": content})

    return chat_display, state


def init_state():
    return {
        "quiz_state": {"quiz_index": 0, "questions": [], "user_answers": []},
        "chat_history": [],
    }


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
