import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Bank Churn AI",
    page_icon="🏦"
)

st.title("🏦 Bank Churn AI")
st.write("Customer retention assistant")


customer_id = st.text_input("Customer ID")

complaint = st.text_area(
    "Customer Complaint",
    height=150
)


if st.button("Analyse Customer"):

    if not customer_id:
        st.warning("Please enter a Customer ID.")

    elif not complaint:
        st.warning("Please enter a complaint.")

    else:

        response = requests.get(
            f"{API_URL}/customers/{customer_id}"
        )

        if response.status_code == 404:
            st.error("Customer not found.")

        elif response.status_code != 200:
            st.error("Error communicating with API.")

        else:

            data = response.json()

            customer = data["customer"]
            churn = data["churn"]

            st.subheader("Customer")

            col1, col2, col3 = st.columns(3)

            col1.metric("Age", customer["age"])
            col2.metric("Balance", f"€{customer['balance']:,.2f}")
            col3.metric(
                "Products",
                customer["num_products"]
            )

            st.subheader("Churn Prediction")

            col1, col2 = st.columns(2)

            col1.metric(
                "Churn Probability",
                f"{churn['churn_probability']:.1%}"
            )

            col2.metric(
                "Risk",
                churn["risk"]
            )

            st.subheader("Complaint")

            st.write(complaint)

            st.info(
                "AI analysis will be added with Microsoft Foundry."
            )