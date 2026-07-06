import os
import sys

from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"


@st.cache_resource
def get_model() -> genai.GenerativeModel:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in the environment.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def build_prompt(topic: str) -> str:
    return f"""
Generate:
1. A deep quote
2. Short explanation
3. Reel hook

Topic: {topic}
""".strip()


def main() -> None:
    st.title("AI Quote Generator")
    st.write("Turn a topic or mood into a quote, a short explanation, and a reel hook.")

    topic = st.text_input("Enter a topic or mood:")

    if st.button("Generate", type="primary"):
        if not topic.strip():
            st.warning("Enter a topic before generating.")
            return

        try:
            model = get_model()
            with st.spinner("Generating your quote..."):
                response = model.generate_content(build_prompt(topic.strip()))
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            return

        if getattr(response, "text", None):
            st.write(response.text)
        else:
            st.warning("The model returned no text response.")


def should_run_with_streamlit() -> bool:
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    if should_run_with_streamlit():
        main()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", sys.argv[0]]
        raise SystemExit(stcli.main())