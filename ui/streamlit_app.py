import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "customer_loaded" not in st.session_state:
    st.session_state["customer_loaded"] = False

if "current_customer_id" not in st.session_state:
    st.session_state["current_customer_id"] = None

if "current_customer" not in st.session_state:
    st.session_state["current_customer"] = None

if "current_churn" not in st.session_state:
    st.session_state["current_churn"] = None

if "current_complaints" not in st.session_state:
    st.session_state["current_complaints"] = []

if "show_history" not in st.session_state:
    st.session_state["show_history"] = False


st.set_page_config(
    page_title="Bank Churn AI",
    page_icon="🏦"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <h1 style="
        color: #4DA3FF;
        font-weight: 700;
        text-align: center;
    ">
        🏦 Bank Churn AI
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div style='text-align: center;'>Customer retention assistant</div>",
    unsafe_allow_html=True
)


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

with st.form("customer_analysis_form", clear_on_submit=True):

    customer_id = st.text_input(
        "Customer ID"
    )

    complaint = st.text_area(
        "Customer Complaint",
        height=150
    )

    submitted = st.form_submit_button(
        "Analyse Customer"
    )


# --------------------------------------------------
# SEARCH CUSTOMER
# --------------------------------------------------

if submitted:

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

            # Store customer information
            st.session_state["current_customer_id"] = customer_id
            st.session_state["current_customer"] = data["customer"]
            st.session_state["current_churn"] = data["churn"]
            st.session_state["current_complaints"] = data.get(
                "complaints",
                []
            )

            st.session_state["customer_loaded"] = True
            st.session_state["show_history"] = False

            # --------------------------------------------------
            # AI ANALYSIS
            # --------------------------------------------------

            analysis_response = requests.post(
                f"{API_URL}/analyze",
                json={
                    "customer_id": customer_id,
                    "complaint": complaint
                }
            )

            if analysis_response.status_code == 200:

                analysis_data = analysis_response.json()

                st.session_state["analysis_data"] = analysis_data

            else:

                st.session_state["analysis_data"] = None


# --------------------------------------------------
# DISPLAY CUSTOMER
# --------------------------------------------------

if st.session_state["customer_loaded"]:

    customer_id = st.session_state["current_customer_id"]
    customer = st.session_state["current_customer"]
    churn = st.session_state["current_churn"]

    # --------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------

    st.subheader(
        f"Customer - {customer_id}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Age",
        customer["age"]
    )

    col2.metric(
        "Balance",
        f"€{customer['balance']:,.2f}"
    )

    col3.metric(
        "Products",
        customer["num_products"]
    )


    # --------------------------------------------------
    # COMPLAINT HISTORY
    # --------------------------------------------------

    if st.button("View Complaint History"):

        st.session_state["show_history"] = True


    if st.session_state["show_history"]:

        complaints = st.session_state["current_complaints"]

        st.markdown(
            "### Complaint History"
        )

        if not complaints:

            st.info(
                "No complaints found for this customer."
            )

        else:

            for complaint_item in complaints:

                st.markdown(
                    f"""
                    **Date:** {complaint_item.get("date", "N/A")}  
                    **Channel:** {complaint_item.get("channel", "N/A")}  
                    **Sentiment:** {complaint_item.get("sentiment", "N/A")}

                    {complaint_item.get("text", "")}
                    """
                )

                st.divider()


    # --------------------------------------------------
    # CHURN PREDICTION
    # --------------------------------------------------

    risk = churn["risk"].upper()

    if risk == "HIGH":

        risk_color = "red"
        risk_icon = "⚠️"

    elif risk == "LOW":

        risk_color = "green"
        risk_icon = ""

    else:

        risk_color = "orange"
        risk_icon = ""


    st.subheader(
        f"{risk_icon} Churn Prediction"
    )

    col1, col2 = st.columns(2)

    col1.markdown(
        f'<div style="font-size: 14px; margin-bottom: 4px;">'
        f'Churn Probability'
        f'</div>'
        f'<div style="font-size: 32px; '
        f'font-weight: bold; '
        f'color: {risk_color};">'
        f'{churn["churn_probability"]:.1%}'
        f'</div>',
        unsafe_allow_html=True
    )

    col2.markdown(
        f'<div style="font-size: 14px; margin-bottom: 4px;">'
        f'Risk'
        f'</div>'
        f'<div style="font-size: 32px; '
        f'font-weight: bold; '
        f'color: {risk_color};">'
        f'{risk}'
        f'</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------

    st.subheader("🤖 AI Analysis")

    analysis_data = st.session_state.get(
        "analysis_data"
    )

    if analysis_data:

        if analysis_data.get("status") == "FAILED":

            st.warning(
                "Customer request was saved, "
                "but the Foundry Agent is currently unavailable."
            )

        elif analysis_data.get("analysis"):

            st.write(
                analysis_data["analysis"]
            )