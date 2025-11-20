import os

import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


# ==============================
# .env の読み込み
# ==============================
load_dotenv()  # 同じフォルダの .env から OPENAI_API_KEY を読み込む


# ==============================
# LLM の準備（旧 LangChain スタイル）
# ==============================
llm = ChatOpenAI(
    model_name="gpt-4o-mini",  # 講座側の指定に合わせてOK
    temperature=0.5,
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

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    # チェーンを実行
    response_text: str = chain.run({"input_text": user_input})
    return response_text


# ==============================
# Streamlit アプリ本体
# ==============================
def main():
    st.set_page_config(
        page_title="Streamlit × LangChain LLMサンプル",
        page_icon="🤖",
        layout="centered",
    )

    st.title("🤖 Streamlit × LangChain LLMサンプル")

    st.markdown(
        """
### 📝 アプリの概要

このアプリでは、画面に入力したテキストをもとに、回答を生成します。 
選択した「専門家」の立場からアドバイスを返してもらうことができます。

### ✅ 使い方

1. まず「どんな専門家に相談するか」をボタンから選びます  
2. 下のテキスト入力欄に、相談したい内容や質問を入力します  
3. 「相談する」ボタンを押すと、回答が画面に表示されます  

"""
    )

    # 専門家の種類（ラジオボタン）
    expert_type = st.radio(
        "相談したい専門家の種類を選んでください：",
        options=["キャリアコーチ", "健康アドバイザー"],
        horizontal=True,
    )

    # 入力フォーム
    user_input = st.text_area(
        label="ここに相談内容 / 質問を入力してください。",
        height=180,
        placeholder="例）40代からAIスキルを身につけるには、どんな勉強方法が良いですか？",
    )

    # ボタン
    if st.button("相談する"):
        if not user_input.strip():
            st.warning("テキストが入力されていません。何か相談内容を入力してください。")
        else:
            with st.spinner("回答を考えています..."):
                answer = ask_llm(user_input=user_input, expert_type=expert_type)

            st.markdown("### 💬 回答")
            st.write(answer)


# ★★★ ここがめちゃくちゃ大事！！ ★★★
if __name__ == "__main__":
    main()
