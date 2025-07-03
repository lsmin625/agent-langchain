import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# ToolExecutor는 LLM이 호출하기로 결정한 도구를 실제로 실행하는 역할을 합니다.
from langchain.agents import ToolExecutor

# OpenAIToolCallParser는 LLM의 응답에서 도구 호출 정보를 파싱합니다.
from langchain_core.output_parsers.openai_tools import OpenAIToolCallParser

# OPENAI_API_KEY 설정 (실제 키로 교체하거나 환경 변수로 설정하세요)
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# 도구 정의 (변경 없음)
@tool
def tranquilizer_watch(_) -> str:
    """범인을 잠재워야 할 때 사용합니다."""
    return "🕶️ 마취 시계를 사용하여 범인을 잠재웁니다."


@tool
def voice_changer_bowtie(_) -> str:
    """다른 사람의 목소리를 흉내 내어 추리를 해야 할 때 사용합니다."""
    return "🎤 음성 변조 나비넥타이를 사용하여 추리를 시작합니다."


@tool
def detective_glasses(_) -> str:
    """무언가를 추적하거나 숨겨진 단서를 찾아야 할 때 사용합니다."""
    return "🔍 범인 추적 안경으로 단서를 찾습니다."


@tool
def soccer_shoes(_) -> str:
    """강력한 물리력으로 장애물을 제거해야 할 때 사용합니다."""
    return "⚽ 킥력 강화 축구화로 장애물을 부숩니다."


# 사용 가능한 도구들을 리스트로 묶습니다.
tools = [tranquilizer_watch, voice_changer_bowtie, detective_glasses, soccer_shoes]

# 1. 프롬프트 개선: 더 이상 LLM에게 '이름만 출력'하라고 지시할 필요가 없습니다.
# 상황을 설명하고 적절한 도구를 '사용'하라고 자연스럽게 요청합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 명탐정 코난입니다. 주어진 상황에 가장 적절한 도구를 사용해서 문제를 해결하세요.",
        ),
        ("human", "{situation}"),
    ]
)

# 2. 모델에 Tool을 바인딩합니다.
# 이렇게 하면 LLM은 자신이 어떤 도구를 사용할 수 있는지 인지하게 됩니다.
llm_with_tools = llm.bind_tools(tools)

# 3. Tool Parser 와 Executor를 사용하여 체인을 구성합니다.
# 이 부분이 기존의 RunnableBranch를 대체합니다.
tool_executor = ToolExecutor(tools=tools)
parser = OpenAIToolCallParser()

# 4. 새로운 파이프라인(체인)
# prompt -> llm_with_tools: LLM이 상황에 맞는 Tool Call을 결정
# parser: LLM의 응답에서 Tool Call 정보를 추출
# tool_executor: 추출된 정보를 바탕으로 실제 도구 함수를 실행
chain = prompt | llm_with_tools | parser | tool_executor
