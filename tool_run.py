import os
import gradio as gr
from dotenv import load_dotenv
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

from langchain_core.messages import AIMessage


# .env 파일 로드 및 API 키 설정
load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI LLM 준비
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


## 단계 - 도구 구성
@tool
def tranquilizer_watch(target: str) -> str:
    """마취 시계: 지정된 대상을 잠재울 필요가 있을 때 사용합니다. 추리 설명 등을 대신할 때 유용합니다.
    Args:
        target (str): 마취시킬 대상의 이름이나 인상착의. 예: '안경 쓴 범인', '유명한 탐정님'
    """
    return f"🕶️ 마취 시계: '{target}'을(를) 성공적으로 마취시켰습니다."


@tool
def voice_changer_bowtie(target: str) -> str:
    """음성 변조 나비넥타이: 다른 사람의 목소리로 추리를 설명하거나, 다른 사람인 척 연기해야 할 때 사용합니다.
    Args:
        target (str): 목소리를 흉내 낼 대상. 예: '브라운 박사님', '유명한 탐정님'
    """
    return f"🎤 음성 변조 나비넥타이: '{target}'의 목소리로 변조를 시작합니다."


@tool
def detective_glasses(target: str) -> str:
    """탐정 안경: 특정 대상을 추적하거나 멀리 있는 것을 확대해서 볼 때 사용합니다. 범인 추적에 필수적입니다.
    Args:
        target (str): 추적하거나 확대할 대상. 예: '범인의 자동차', '먼 곳의 단서'
    """
    return f"🔍 탐정 안경: '{target}'에 대한 추적 및 확대 기능을 활성화합니다."


@tool
def soccer_shoes(target: str) -> str:
    """킥력 강화 축구화: 강력한 힘으로 무언가를 걷어차 범인을 제압하거나 위기 상황을 탈출할 때 사용합니다.
    Args:
        target (str): 강하게 찰 대상. 예: '범인을 위협할 돌멩이', '막다른 길의 문'
    """
    return f"⚽ 킥력 강화 축구화: '{target}'을(를) 향해 강력한 킥을 준비합니다!"


# 도구 목록 및 LLM 바인딩
tools = [tranquilizer_watch, voice_changer_bowtie, detective_glasses, soccer_shoes]
llm_with_tools = llm.bind_tools(tools)


## 단계 - 프롬프트
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 명탐정 코난입니다. 주어진 상황을 해결하기 위해 당신이 가진 도구들을 가장 적절하게 사용하세요. 상황에 따라 여러 도구를 동시에 사용할 수도 있습니다.",
        ),
        ("human", "{situation}"),
    ]
)


## 단계 - 도구 검색

# 실행할 도구를 이름으로 쉽게 찾을 수 있도록 맵을 생성
tool_map = {tool.name: tool for tool in tools}


def tool_executor(model_output: AIMessage) -> Any:
    """LLM의 출력을 받아, tool_calls가 있으면 해당 도구를 실행하고 결과를 반환합니다."""
    if not isinstance(model_output, AIMessage) or not model_output.tool_calls:
        # LLM이 도구를 사용하지 않고 일반 메시지로 답변한 경우, 내용을 그대로 반환
        return model_output.content

    results = []
    # LLM이 호출한 모든 도구를 순차적으로 실행
    for tool_call in model_output.tool_calls:
        tool_to_run = tool_map.get(tool_call["name"])
        if tool_to_run:
            # .invoke 메소드는 인자 딕셔너리를 자동으로 처리
            observation = tool_to_run.invoke(tool_call["args"])
            results.append(observation)
        else:
            results.append(f"오류: '{tool_call['name']}' 도구를 찾을 수 없습니다.")

    return "\n".join(str(res) for res in results)


# 단계 - LCEL 체인 구성
chain = prompt | llm_with_tools | tool_executor


# 단계 - Gradio 인터페이스
def handle_tool_selection(user_input: str) -> str:
    """Gradio 입력값을 받아 개선된 체인을 실행하고 결과를 반환합니다."""
    if not user_input:
        return "상황을 입력해주세요."
    return chain.invoke({"situation": user_input})


with gr.Blocks() as demo:
    gr.Markdown("## 🕵️ 명탐정 코난 도구 추천기")
    gr.Markdown("상황을 입력하면 코난이 사용할 적절한 도구를 추천해 드립니다.")

    input_box = gr.Textbox(
        label="상황 설명", placeholder="예: 용의자를 조용히 기절시키고 싶어요"
    )
    output_box = gr.Textbox(label="추천 도구", lines=4)

    input_box.submit(handle_tool_selection, inputs=input_box, outputs=output_box)

demo.launch()
