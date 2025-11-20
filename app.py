from dotenv import load_dotenv
import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# 環境変数の読み込み（ローカルの .env 用）
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

    # プロンプトテンプレート（system ＋ human）
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "{input_text}"),
        ]
    )

    # LLMChain を作成
    chain = LLMChain(llm=llm, prompt=prompt)

    # チェーンに入力して回答を取得
    response = chain.run({"input_text": user_input})

    return response


# ==============================
# Streamlit アプリ本体
# ==============================
st.set_page_config(
    page_title="Streamlit × LangChain LLMサンプル",
    page_icon="🤖",
)

st.title("🤖 Streamlit × LangChain LLMサンプル")

st.markdown(
    """
### 📄 アプリの概要
このアプリでは、画面に入力したテキストを **LangChain** を使って LLM に渡し、
選択した「専門家」の立場からアドバイスを返してもらうことができます。

### ✅ 使い方
1. まず **「どんな専門家に相談するか」** をラジオボタンから選びます  
2. 下のテキスト入力欄に、相談したい内容や質問を書きます  
3. **「LLMに相談する」** ボタンを押すと、LLMからの回答が画面に表示されます  

※このアプリを動かすには、`.env` ファイルなどで `OPENAI_API_KEY` を設定しておいてください。
"""
)

# 専門家の種類を選ぶラジオボタン
expert_type = st.radio(
    "相談したい専門家の種類を選んでください：",
    ("キャリアコーチ", "健康アドバイザー"),
)

# ユーザー入力欄
user_text = st.text_area(
    "ここに相談内容 / 質問を書いてください。",
    height=200,
    placeholder="例）40代からAIスキルを身につけるには、どんな勉強方法が良いですか？",
)

# ボタン押下で LLM に問い合わせ
if st.button("LLMに相談する"):
    if user_text.strip() == "":
        st.warning("まずは相談内容を入力してください。")
    else:
        with st.spinner("回答を生成しています..."):
            answer = ask_llm(user_text, expert_type)

        st.subheader("💡 LLMからの回答")
        st.write(answer)
