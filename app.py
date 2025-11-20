from dotenv import load_dotenv
import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# .env から環境変数を読み込む
load_dotenv()

# ===== LLM（OpenAI GPT）の設定 =====
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
)


# ==============================
# LLM に質問する関数
# ==============================
def ask_llm(user_input: str, expert_type: str) -> str:
    """入力テキストと専門家の種類から、LLMの回答を返す"""

    # ラジオボタンの選択によって system メッセージを切り替え
    if expert_type == "キャリアコーチ":
        system_template = (
            "あなたは日本語で丁寧に回答するキャリアコーチです。"
            "相談者の気持ちに寄り添いながら、仕事・転職・スキルアップに関する具体的なアドバイスを3つ提示してください。"
        )
    elif expert_type == "健康アドバイザー":
        system_template = (
            "あなたは日本語で丁寧に回答する健康アドバイザーです。"
            "睡眠・食事・運動など、日常生活の中で無理なく続けられる健康習慣を、相談内容に合わせて3つ提示してください。"
        )
    else:
        system_template = "あなたは日本語で丁寧に回答する親切なアドバイザーです。"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "{input_text}"),
        ]
    )

    # プロンプトをメッセージに変換して LLM を呼び出す
    messages = prompt.format_messages(input_text=user_input)
    response = llm.invoke(messages)

    return response.content


# ==============================
# Streamlit 画面
# ==============================
st.set_page_config(page_title="Streamlit × LangChain LLMサンプル")

st.title("🧠 Streamlit × LangChain LLMサンプル")

st.subheader("📄 アプリの概要")
st.write(
    """
このアプリでは、画面に入力したテキストを LangChain を使って LLM に渡し、
選択した「専門家」の立場からアドバイスを返してもらうことができます。
"""
)

st.subheader("✅ 使い方")
st.markdown(
    """
1. まず「どんな専門家に相談するか」をラジオボタンから選びます  
2. 下のテキスト入力欄に、相談したい内容や質問を入力します  
3. 「LLMに相談する」ボタンを押すと、LLMからの回答が画面に表示されます  

※ このアプリを動かすには、`.env` ファイルなどで `OPENAI_API_KEY` を設定しておいてください。
"""
)

# 専門家の種類を選ぶラジオボタン
expert_type = st.radio(
    "相談したい専門家の種類を選んでください：",
    ("キャリアコーチ", "健康アドバイザー"),
)

# テキスト入力欄
user_input = st.text_area(
    "ここに相談内容／質問を入力してください。",
    placeholder="例）40代からAIスキルを身につけるには、どんな勉強方法が良いですか？",
    height=150,
)

# ボタンを押したら LLM に問い合わせ
if st.button("💬 LLMに相談する"):
    if not user_input.strip():
        st.warning("相談内容を入力してください。")
    else:
        with st.spinner("LLMに相談中です..."):
            answer = ask_llm(user_input, expert_type)
        st.subheader("📝 回答")
        st.write(answer)

