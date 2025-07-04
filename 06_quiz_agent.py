import gradio as gr
import random, json, os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# 퀴즈 파일 및 출제 문항 개수 지정
QUIZ_FILE = "data/conan_quiz.json"
QUIZ_COUNT = 3

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


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
    processed = user_input.strip()  # 사용자 입력 앞뒤 공백 제거

    if q["type"] == "multiple_choice":
        try:
            sel = int(processed) - 1  # 사용자 입력을 선다형 인덱스로 변환 (1 작은 수)
            if 0 <= sel < len(q["choices"]):  # 인덱스 유효 범위 확인
                processed = q["choices"][sel]
        except:
            pass  # 입력 상태 유지

    state[
        "user_answers"
    ].append(  # 사용자 응답 결과를 기록하는 리스트에 새로운 항목을 추가
        {
            "question_text": q["question"],  # 퀴즈 문항 질문
            "user_response": processed,  # 사용자 답변
            "is_correct": False,  # 정답 여부는 아직 채점 전이므로 일단 False로 저장
            "correct_answer": str(q["answer"]),  # 정답은 문자열로 변환해서 저장
        }
    )
    state["quiz_index"] += 1  # 다음 문제로 넘어가기 위해 인덱스를 1 증가
    return state


# Agent에게 전달할 채점 데이터 생성
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


# Agent에게 부여할 역할 및 지침 정의
agent_kwargs = {
    "system_message": "당신은 '명탐정 코난' 퀴즈의 전문 채점관입니다. 주어진 문제, 정답, 사용자 답변을 바탕으로 채점해주세요. 각 문제에 대해 정답 여부를 판단하고 친절한 해설을 덧붙여주세요. 모든 채점이 끝나면, 마지막에는 '총점: X/Y' 형식으로 최종 점수를 반드시 요약해서 보여줘야 합니다."
}

# 채점 Agent 초기화
grading_agent = initialize_agent(
    tools=[],  # 이 시나리오에서는 외부 도구가 필요 없습니다.
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  # Agent의 작동 과정을 확인하려면 True로 설정
    agent_kwargs=agent_kwargs,
    handle_parsing_errors=True,  # 파싱 오류 발생 시 자연스럽게 대처하도록 설정
)


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

        # Agent 호출
        raw_result = grading_agent.invoke(
            {"input": grading_input_data, "chat_history": []}
        )["output"]

        # [수정된 부분] Agent가 반환한 딕셔너리(또는 JSON 문자열)를
        # 사용자가 보기 좋은 형태의 단일 문자열로 변환합니다.
        try:
            # 결과가 JSON 문자열일 경우를 대비해 파싱 시도
            if isinstance(raw_result, str):
                result_data = json.loads(raw_result)
            else:
                result_data = raw_result  # 이미 딕셔너리인 경우

            # 채점 결과를 바탕으로 문자열 보고서 생성
            report_parts = ["채점이 완료되었습니다! 📝\n"]
            for i, res in enumerate(result_data.get("results", [])):
                is_correct_text = "정답" if res.get("is_correct") else "오답"
                report_parts.append(f"--- 문제 {i+1} ---")
                report_parts.append(f"문제: {res.get('question', '질문 없음')}")
                report_parts.append(f"정답: {res.get('correct_answer', '정답 없음')}")
                report_parts.append(
                    f"제출한 답변: {res.get('user_answer', '답변 없음')}"
                )
                report_parts.append(f"결과: {is_correct_text}")
                report_parts.append(f"해설: {res.get('explanation', '')}\n")

            report_parts.append(
                f"**총점: {result_data.get('total_score', '점수 없음')}**"
            )
            final_report = "\n".join(report_parts)

        except (json.JSONDecodeError, TypeError, AttributeError):
            # 만약 결과가 예상된 딕셔너리 형식이 아니면, 받은 그대로 출력
            final_report = str(raw_result)

        messages.append({"role": "assistant", "content": final_report})

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
    gr.Markdown("### 🕵️ 명탐정 코난 매니아 판별기")

    chatbot = gr.Chatbot(
        label="명탐정 코난 퀴즈 챗봇",
        height=400,
        avatar_images=("data/avatar_user.png", "data/avatar_conan.png"),
        type="messages",
    )

    txt = gr.Textbox(placeholder="'퀴즈 시작'을 입력해보세요!", show_label=False)
    state = gr.State(init_state())

    txt.submit(chat_fn, inputs=[txt, state], outputs=[chatbot, state])
    txt.submit(lambda: "", None, txt)

demo.launch()
