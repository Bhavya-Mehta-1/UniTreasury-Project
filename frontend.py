import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.title("Macro Variable Selector")

if "basket" not in st.session_state:
    st.session_state.basket = []

query = st.text_input("Search for a macro variable (e.g. Oil Price):")

results = []

if query:
    response = requests.get(f"{API}/search", params={"query": query})
    results = response.json()
    st.dataframe(results)

options = [r["code"] for r in results]
chosen = st.multiselect("Pick the variables to analyse", options)

if st.button("Add to basket"):
    for code in chosen:
        if code not in st.session_state.basket:
            st.session_state.basket.append(code)

if st.button("Clear basket"):
    st.session_state.basket = []

st.subheader("Your basket")
st.write(st.session_state.basket)

if len(st.session_state.basket) >= 2:
    target = st.selectbox("Target variable (the one to explain)", st.session_state.basket)

    if st.button("Analyze"):
        codes = ",".join(st.session_state.basket)
        resp = requests.get(f"{API}/analyze", params={"codes": codes, "target": target})
        st.session_state.result = resp.json()
        st.session_state.pop("explanation", None)

if "result" in st.session_state:
    result = st.session_state.result

    st.subheader("Selected variables")
    st.write(result["selected_variables"])

    st.subheader("Model fit")
    st.write(result["model"])

    st.subheader("Coefficients")
    st.dataframe(result["coefficients"])

    st.subheader("Warnings")
    for f in result["flags"]:
        st.warning(f)

    if st.button("Explain in plain English"):
        resp = requests.post(f"{API}/explain", json=result)
        st.session_state.explanation = resp.json()["explanation"]

    if "explanation" in st.session_state:
        st.subheader("Use GROQ AI for a simpler explanation")
        st.write(st.session_state.explanation)

    with st.expander("Raw output"):
        st.json(result)