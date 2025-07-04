import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.tools import tool

# .env 파일 로드
load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")
print(f"{open_api_key[:9]}***")

# OpenAI LLM 준비
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print(llm.model_name)


@tool
def tranquilizer_watch() -> str:
    """마취 시계: 용의자를 조용히 기절시킬 때 사용하는 도구입니다."""
    return "🕶️ 마취 시계: 용의자를 조용히 기절시킬 때 사용하세요."


@tool
def voice_changer_bowtie() -> str:
    """변조 나비넥타이: 다른 사람의 목소리를 흉내 낼 때 사용하는 도구입니다."""
    return "🎤 변조 나비넥타이: 다른 사람의 목소리를 흉내 낼 때 사용하세요."


@tool
def detective_glasses() -> str:
    """탐정 안경: 멀리 있는 대상을 확대하거나 적외선으로 추적할 때 사용하는 도구입니다."""
    return "🔍 탐정 안경: 멀리 있는 대상을 확대하거나 적외선으로 추적할 때 사용하세요."


@tool
def soccer_shoes() -> str:
    """킥력 강화 축구화: 멀리 있는 물체를 정확히 차거나 위협할 때 사용하는 도구입니다."""
    return "⚽ 킥력 강화 축구화: 멀리 있는 물체를 정확히 차거나 위협할 때 사용하세요."


# 도구 매핑
tool_map = {
    "마취 시계": tranquilizer_watch,
    "변조 나비넥타이": voice_changer_bowtie,
    "탐정 안경": detective_glasses,
    "킥력 강화 축구화": soccer_shoes,
}

prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", "당신은 명탐정 코난입니다. 주어진 상황에 가장 적절한 도구를 사용해서 문제를 해결하세요."),
        # ("human", "{situation}")
        (
            "system",
            """당신은 '명탐정 코난'의 도구 전문가입니다.
사용자의 상황 설명을 보고, 아래 목록 중에서 필요한 도구를 모두 선택하여, 한 줄에 하나씩 도구 이름만 정확하게 나열해 주세요.

- 마취 시계
- 변조 나비넥타이
- 탐정 안경
- 킥력 강화 축구화

다른 말은 절대 하지 말고, 필요한 도구의 이름만 나열하세요.""",
        ),
        ("human", "{situation}"),
    ]
)


# 여러 도구를 실행하고 결과를 조합하는 함수
def run_and_combine_tools(tool_names_str: str) -> str:
    """LLM이 반환한 여러 줄의 도구 이름 문자열을 받아, 각 도구를 실행하고 결과를 합칩니다."""
    # LLM 출력에서 앞뒤 공백을 제거하고, 줄바꿈을 기준으로 나누어 리스트 생성
    tool_names = [
        name.strip() for name in tool_names_str.strip().split("\n") if name.strip()
    ]

    if not tool_names:
        return "🤔 추천할 도구가 없습니다."

    results = []
    for name in tool_names:
        tool_fn = tool_map.get(name)
        if tool_fn:
            # 각 Tool 객체를 인자 없이 실행
            results.append(tool_fn.invoke({}))
        else:
            # LLM이 목록에 없는 도구를 생성한 경우를 대비
            results.append(f"❌ '{name}'은(는) 알 수 없는 도구입니다.")

    # 모든 결과를 줄바꿈으로 합쳐서 하나의 문자열로 반환
    return "\n".join(results)


# [수정] LCEL 파이프라인 구성 (새로운 실행 함수 연결)
tool_selector_chain = (
    prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(run_and_combine_tools)  # 새로운 함수를 연결
)


# Gradio 처리 함수
def handle_tool_selection(user_input):
    return tool_selector_chain.invoke({"situation": user_input})


with gr.Blocks() as demo:
    gr.Markdown("## 🕵️ 명탐정 코난 도구 추천기")
    gr.Markdown("상황을 입력하면 코난이 사용할 적절한 도구를 추천해 드립니다.")

    input_box = gr.Textbox(
        label="상황 설명", placeholder="예: 용의자를 조용히 기절시키고 싶어요"
    )
    output_box = gr.Textbox(label="추천 도구", lines=4)

    input_box.submit(handle_tool_selection, inputs=input_box, outputs=output_box)

demo.launch()
