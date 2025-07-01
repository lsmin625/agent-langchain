import os
import json
import random
import logging

from dotenv import load_dotenv
from typing import Annotated, List, Dict, Any, Union
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# --- 설정 및 초기화 ---

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 환경 변수 로딩
load_dotenv()

# 상수 정의
QUIZ_FILE_PATH = "conan_quiz.json"
NUM_QUESTIONS_TO_SELECT = 3
LLM_MODEL_NAME = "openai:gpt-4o"

# --- 데이터 모델 정의 ---


class Question(TypedDict):
    """퀴즈 문제의 구조를 정의합니다."""

    question: str
    type: str  # 예: "multiple_choice", "short_answer"
    choices: List[str]  # 객관식일 경우
    answer: Union[str, int]  # 정답 (객관식은 선택지 인덱스, 주관식은 텍스트)


class UserAnswer(TypedDict):
    """사용자 답변의 구조를 정의합니다."""

    question_text: str
    user_response: str
    is_correct: bool  # 채점 후 추가될 필드
    correct_answer: str  # 채점 후 추가될 필드


class QuizState(TypedDict):
    """LangGraph 상태를 정의합니다."""

    messages: Annotated[List[BaseMessage], add_messages]
    quiz_index: int
    user_answers: List[UserAnswer]
    questions: List[Question]


# --- 퀴즈 데이터 로딩 ---


def load_questions(file_path: str, num_questions: int) -> List[Question]:
    """
    JSON 파일에서 퀴즈 문제를 로드하고 지정된 수만큼 랜덤하게 선택합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
        if not all_questions:
            raise ValueError("퀴즈 파일에 문제가 포함되어 있지 않습니다.")
        if len(all_questions) < num_questions:
            logger.warning(
                f"요청된 문제 수({num_questions})가 전체 문제 수({len(all_questions)})보다 많습니다. 모든 문제를 사용합니다."
            )
            return all_questions
        selected = random.sample(all_questions, num_questions)
        logger.info(
            f"랜덤으로 선택된 문제 {len(selected)}개: {[q['question'] for q in selected]}"
        )
        return selected
    except FileNotFoundError:
        logger.error(f"퀴즈 파일이 존재하지 않습니다: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(
            f"퀴즈 파일을 디코딩하는 데 실패했습니다. JSON 형식을 확인해주세요: {file_path}"
        )
        raise
    except Exception as e:
        logger.error(f"문제 로딩 중 예기치 않은 오류 발생: {e}")
        raise


# --- LLM 초기화 ---


def initialize_llm(model_name: str):
    """LLM 모델을 초기화합니다."""
    try:
        llm = init_chat_model(model_name)
        logger.info(f"LLM 모델 초기화 완료: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"LLM 모델 초기화 실패: {e}")
        raise


llm = initialize_llm(LLM_MODEL_NAME)

# --- LangGraph 노드 정의 ---


def ask_question_node(state: QuizState) -> Dict[str, Any]:
    """
    현재 퀴즈 인덱스에 해당하는 질문을 사용자에게 제시하고 상태를 업데이트합니다.
    """
    idx = state["quiz_index"]
    questions = state["questions"]

    if idx >= len(questions):
        # 모든 질문이 소진되었을 경우를 대비한 방어 코드 (조건부 엣지에서 처리되지만 안전을 위해)
        logger.warning("모든 질문이 소진되었는데 ask_question_node가 호출되었습니다.")
        return {"messages": []}

    question_data = questions[idx]
    q_text = f"문제 {idx + 1}. {question_data['question']}"

    if question_data["type"] == "multiple_choice":
        choices = [
            f"{i + 1}. {choice}" for i, choice in enumerate(question_data["choices"])
        ]
        q_text += "\n선택지:\n" + "\n".join(choices)

    logger.info(f"문제 출제: {q_text.splitlines()[0]}...")  # 첫 줄만 로깅
    print(f"\n--- 문제 {idx + 1} ---")
    print(q_text)
    print("--------------------")

    return {"messages": [("system", q_text)]}


def record_answer_node(state: QuizState) -> Dict[str, Any]:
    """
    사용자의 답변을 입력받고, 객관식인 경우 숫자를 선택지로 변환하여 상태에 기록합니다.
    다음 질문으로 넘어가기 위해 quiz_index를 증가시킵니다.
    """
    idx = state["quiz_index"]
    question_data = state["questions"][idx]

    user_input = input("답변 (객관식은 번호, 주관식은 직접 입력): ").strip()
    processed_answer = user_input

    # 객관식인 경우, 입력된 숫자를 실제 선택지 텍스트로 변환
    if question_data["type"] == "multiple_choice":
        try:
            answer_index = int(user_input) - 1
            if 0 <= answer_index < len(question_data["choices"]):
                processed_answer = question_data["choices"][answer_index]
                logger.info(f"객관식 답변 변환: '{user_input}' -> '{processed_answer}'")
            else:
                print(
                    "⚠️ 잘못된 선택지 번호입니다. 입력하신 내용 그대로 답변으로 기록합니다."
                )
                logger.warning(f"잘못된 선택지 번호 입력: '{user_input}'")
        except ValueError:
            print(
                "⚠️ 숫자가 아닌 값이 입력되었습니다. 입력하신 내용 그대로 답변으로 기록합니다."
            )
            logger.warning(f"객관식에 비숫자 값 입력: '{user_input}'")

    # 사용자 답변 기록
    new_user_answer: UserAnswer = {
        "question_text": question_data["question"],
        "user_response": processed_answer,
        "is_correct": False,  # 초기값
        "correct_answer": str(
            question_data["answer"]
        ),  # 초기값 (채점 시 실제 정답으로 업데이트)
    }

    updated_user_answers = state["user_answers"] + [new_user_answer]
    next_quiz_index = state["quiz_index"] + 1

    logger.info(f"사용자 답변 기록됨 (문제 {idx + 1}): '{processed_answer}'")
    return {
        "messages": [("user", processed_answer)],
        "user_answers": updated_user_answers,
        "quiz_index": next_quiz_index,
    }


def evaluate_node(state: QuizState) -> Dict[str, Any]:
    """
    저장된 사용자 답변과 정답을 비교하여 채점하고, LLM을 활용하여 상세 피드백을 제공합니다.
    """
    grading_prompt_parts = [
        "당신은 퀴즈 채점관입니다. 다음은 사용자가 응답한 퀴즈입니다.",
        "각 문항에 대해 사용자의 답변이 정답인지 여부를 '정답' 또는 '오답'으로 명확히 구분하여 평가해주세요.",
        "이후 각 문제에 대해 간략한 해설 또는 피드백을 제공해주세요.",
        "전체 퀴즈에 대한 총점을 계산하여 마지막에 '총점: X/Y' 형식으로 알려주세요. 각 문제는 1점입니다.",
    ]

    for i, (q_data, ua_data) in enumerate(
        zip(state["questions"], state["user_answers"])
    ):
        grading_prompt_parts.append(f"\n--- 문제 {i + 1} ---")
        grading_prompt_parts.append(f"문제: {q_data['question']}")
        if q_data["type"] == "multiple_choice" and "choices" in q_data:
            grading_prompt_parts.append(f"선택지: {', '.join(q_data['choices'])}")
        grading_prompt_parts.append(f"정답: {q_data['answer']}")
        grading_prompt_parts.append(f"사용자 답변: {ua_data['user_response']}")

    grading_prompt = "\n".join(grading_prompt_parts)

    logger.info("LLM으로 채점 요청 전송...")
    # LLM 호출 시 BaseMessage 리스트로 전달
    llm_response: BaseMessage = llm.invoke([("user", grading_prompt)])

    grading_result_content = llm_response.content

    print("\n" + "=" * 30)
    print("      ✨ 퀴즈 결과 및 채점 ✨")
    print("=" * 30)
    print(grading_result_content)  # LLM의 응답 내용을 바로 출력합니다.
    print("=" * 30 + "\n")

    logger.info("채점 완료 및 결과 출력.")
    return {"messages": [("assistant", grading_result_content)]}


# --- LangGraph 빌드 ---


def create_quiz_graph():
    """LangGraph를 정의하고 빌드합니다."""
    graph_builder = StateGraph(QuizState)

    # 노드 추가
    graph_builder.add_node("ask", ask_question_node)
    graph_builder.add_node("answer", record_answer_node)
    graph_builder.add_node("grade", evaluate_node)

    # 엣지 정의
    graph_builder.add_edge(START, "ask")  # 시작 노드에서 질문 노드로
    graph_builder.add_edge("ask", "answer")  # 질문 노드에서 답변 기록 노드로

    # 조건부 엣지: 모든 질문에 답했으면 채점 노드로, 아니면 다음 질문 노드로
    graph_builder.add_conditional_edges(
        "answer",
        lambda state: (
            "grade" if state["quiz_index"] >= len(state["questions"]) else "ask"
        ),
    )

    # 최종 상태 노드 설정
    graph_builder.set_finish_point("grade")

    logger.info("LangGraph 빌드 완료.")
    return graph_builder.compile()


# --- 메인 실행 ---

if __name__ == "__main__":
    try:
        # 퀴즈 문제 로드
        selected_questions = load_questions(QUIZ_FILE_PATH, NUM_QUESTIONS_TO_SELECT)

        # LangGraph 생성
        quiz_graph = create_quiz_graph()

        # 초기 상태 정의
        initial_state: QuizState = {
            "messages": [],
            "quiz_index": 0,
            "user_answers": [],
            "questions": selected_questions,
        }

        logger.info("퀴즈 시작!")
        # LangGraph 실행 (스트림 모드)
        for event in quiz_graph.stream(initial_state):
            # 각 이벤트 발생 시 특별한 처리가 필요 없다면 pass
            # 디버깅을 위해 각 노드의 출력만 보여줍니다.
            for key, value in event.items():
                if key != "__end__":  # 종료 이벤트는 출력하지 않음
                    logger.debug(f"노드 실행: {key}, 출력: {value}")

        logger.info("퀴즈가 성공적으로 종료되었습니다.")

    except Exception as e:
        logger.critical(f"퀴즈 실행 중 치명적인 오류 발생: {e}", exc_info=True)
        print("\n🚫 퀴즈 실행 중 오류가 발생했습니다. 로그를 확인해주세요.")
