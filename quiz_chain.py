import gradio as gr
import random, json, os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

QUIZ_FILE = "conan_quiz.json"
QUIZ_COUNT = 3

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# 퀴즈 로딩
def load_quiz():
    with open(QUIZ_FILE, "r", encoding="utf-8") as f:
        all_q = json.load(f)
    return random.sample(all_q, QUIZ_COUNT)


# 문제 텍스트 출력
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


# 채점용 프롬프트 생성
def build_grading_prompt(state):
    parts = [
        "당신은 퀴즈 채점관입니다. 사용자 답변을 정답 여부로 판단하고 각 문제에 피드백을 제공해주세요.",
        "마지막에는 '총점: X/Y' 형식으로 출력해주세요.",
    ]
    for i, (q, a) in enumerate(zip(state["questions"], state["user_answers"])):
        parts.append(f"\n문제 {i+1}: {q['question']}")
        if q["type"] == "multiple_choice":
            parts.append(f"선택지: {', '.join(q['choices'])}")
        parts.append(f"정답: {q['answer']}")
        parts.append(f"사용자 답변: {a['user_response']}")
    return "\n".join(parts)


# LCEL 채점 체인
grade_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "채점관으로서 정답 판단 및 피드백을 제공해주세요."),
            ("user", "{grading_input}"),
        ]
    )
    | llm
    | StrOutputParser()
)


# 퀴즈 시작 요청 처리
def handle_quiz_start(user_input, quiz_state, messages):
    quiz_state["questions"] = load_quiz()
    quiz_state["quiz_index"] = 0
    quiz_state["user_answers"] = []
    qtext = get_question(quiz_state)
    messages.append([user_input, qtext])
    return quiz_state, messages


# 퀴즈가 이미 끝난 경우
def handle_quiz_already_done(user_input, messages):
    messages.append(
        [
            user_input,
            "퀴즈가 이미 종료되었습니다. 다시 시작하려면 '퀴즈 시작'이라고 입력하세요.",
        ]
    )
    return messages


# 사용자 답변 처리
def handle_user_answer(user_input, quiz_state, messages):
    quiz_state = update_state(quiz_state, user_input)

    if quiz_state["quiz_index"] < len(quiz_state["questions"]):
        qtext = get_question(quiz_state)
        messages.append([user_input, qtext])
    else:
        prompt = build_grading_prompt(quiz_state)
        result = grade_chain.invoke({"grading_input": prompt})
        messages.append([user_input, result])

    return quiz_state, messages


# 메인 챗봇 처리 함수
def chat_fn(user_input, state):
    user_input_lower = user_input.strip().lower()
    messages = state["chat_history"]
    quiz_state = state["quiz_state"]

    if quiz_state["questions"] == []:
        if user_input_lower in ["퀴즈", "퀴즈 시작"]:
            quiz_state, messages = handle_quiz_start(user_input, quiz_state, messages)
        else:
            messages.append(
                [
                    user_input,
                    "'퀴즈' 또는 '퀴즈 시작'이라고 입력하면 퀴즈를 시작합니다.",
                ]
            )
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
with gr.Blocks() as demo:
    gr.Markdown("## 🕵️ 명탐정 코난 매니아 판별기")
    chatbot = gr.Chatbot(label="명탐정 코난 퀴즈 챗봇", height=400)
    txt = gr.Textbox(placeholder="'퀴즈 시작'을 입력해보세요!", show_label=False)
    state = gr.State(init_state())

    txt.submit(chat_fn, inputs=[txt, state], outputs=[chatbot, state])
    txt.submit(lambda: "", None, txt)

demo.launch()
