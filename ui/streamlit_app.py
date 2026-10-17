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

if "current_retention_requests" not in st.session_state:
    st.session_state["current_retention_requests"] = []

if "show_retention_requests" not in st.session_state:
    st.session_state["show_retention_requests"] = False

if "selected_retention_request_id" not in st.session_state:
    st.session_state["selected_retention_request_id"] = None

if "retention_requests_page" not in st.session_state:
    st.session_state["retention_requests_page"] = 1

RETENTION_REQUESTS_PAGE_SIZE = 5


st.set_page_config(
    page_title="Bank Churn AI",
    page_icon="🏦"
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    .st-key-customer_analysis_form [data-testid="stHorizontalBlock"] {
        gap: 15px !important;
    }
    .st-key-retention_requests_list [data-testid="stVerticalBlock"] {
        gap: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def clear_selected_retention_request():
    st.session_state["selected_retention_request_id"] = None


@st.dialog(
    "Retention Request Details",
    on_dismiss=clear_selected_retention_request
)
def show_retention_request_details(request_item):
    action_type = request_item.get("action_type", "Unknown")
    if action_type == "AI_ANALYSIS":
        action_label = "AI analysis (interaction record)"
    else:
        action_label = action_type.replace("_", " ").title()

    st.markdown(
        f"### {action_label} · {request_item.get('status', 'Unknown')}"
    )

    detail_col1, detail_col2 = st.columns(2)
    detail_col1.write(f"**Request ID:** {request_item.get('id', 'N/A')}")
    detail_col2.write(f"**Currency:** {request_item.get('currency', 'N/A')}")
    detail_col1.write(f"**Created:** {request_item.get('created_at', 'N/A')}")
    detail_col2.write(f"**Approved:** {request_item.get('aproved_at', 'N/A')}")
    detail_col1.write(f"**Executed:** {request_item.get('executed_at', 'N/A')}")
    detail_col2.write(
        f"**Segment / Risk:** "
        f"{request_item.get('segment', 'N/A')} / "
        f"{request_item.get('risk', 'N/A')}"
    )

    if request_item.get("reason"):
        st.write(f"**Reason:** {request_item['reason']}")

    if request_item.get("description"):
        st.write("**Description**")
        st.write(request_item["description"])

    if st.button("Close", key=f"close_retention_request_{request_item.get('id')}"):
        clear_selected_retention_request()
        st.rerun()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <h1 style="
        color: #4DA3FF;
        font-weight: 500;
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
        height=75
    )

    analyse_col, retention_col = st.columns(2, gap="small")

    with analyse_col:
        submitted = st.form_submit_button(
            "Analyse Customer"
        )

    with retention_col:
        view_retention_requests_submitted = st.form_submit_button(
            "View Retention Requests"
        )


if view_retention_requests_submitted:

    if not customer_id:

        st.warning("Please enter a Customer ID.")

    else:

        try:
            customer_response = requests.get(
                f"{API_URL}/customers/{customer_id}",
                timeout=10
            )

            if customer_response.status_code == 404:

                st.error("Customer not found.")

            elif customer_response.status_code != 200:

                st.error("Error loading customer information.")

            else:

                retention_response = requests.get(
                    f"{API_URL}/customers/{customer_id}/retention-requests",
                    timeout=10
                )
                retention_response.raise_for_status()

                data = customer_response.json()
                if st.session_state["current_customer_id"] != customer_id:
                    st.session_state["analysis_data"] = None

                st.session_state["current_customer_id"] = customer_id
                st.session_state["current_customer"] = data["customer"]
                st.session_state["current_churn"] = data["churn"]
                st.session_state["current_complaints"] = data.get("complaints", [])
                st.session_state["customer_loaded"] = True
                st.session_state["show_history"] = False
                st.session_state["current_retention_requests"] = (
                    retention_response.json().get("retention_requests", [])
                )
                st.session_state["show_retention_requests"] = True
                st.session_state["selected_retention_request_id"] = None
                st.session_state["retention_requests_page"] = 1

        except requests.RequestException as error:
            st.error(f"Could not load retention requests: {error}")


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
            st.session_state["current_retention_requests"] = []
            st.session_state["show_retention_requests"] = False
            st.session_state["selected_retention_request_id"] = None
            st.session_state["retention_requests_page"] = 1

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


    if st.session_state["show_retention_requests"]:

        retention_requests = st.session_state["current_retention_requests"]

        st.markdown(
            "### Retention Requests"
        )

        if not retention_requests:

            st.info(
                "No retention requests found for this customer."
            )

        else:

            total_pages = max(
                1,
                (len(retention_requests) + RETENTION_REQUESTS_PAGE_SIZE - 1)
                // RETENTION_REQUESTS_PAGE_SIZE
            )
            current_page = min(
                max(st.session_state["retention_requests_page"], 1),
                total_pages
            )
            st.session_state["retention_requests_page"] = current_page
            start_index = (current_page - 1) * RETENTION_REQUESTS_PAGE_SIZE
            page_requests = retention_requests[
                start_index:start_index + RETENTION_REQUESTS_PAGE_SIZE
            ]

            with st.container(key="retention_requests_list"):
                header = st.columns([2.1, 1.0, 1.6, 1.3, 0.55])
                header[0].markdown("**Request**")
                header[1].markdown("**Status**")
                header[2].markdown("**Created**")
                header[3].markdown("**Segment / Risk**")
                header[4].markdown("**Info**")

                for request_item in page_requests:

                    action_type = request_item.get("action_type", "Unknown")
                    if action_type == "AI_ANALYSIS":
                        action_label = "AI analysis"
                    else:
                        action_label = action_type.replace("_", " ").title()

                    row = st.columns(
                        [2.1, 1.0, 1.6, 1.3, 0.55],
                        vertical_alignment="center"
                    )
                    row[0].write(action_label)
                    row[1].write(request_item.get("status", "Unknown"))
                    row[2].write(str(request_item.get("created_at", "N/A"))[:16])
                    row[3].write(
                        f"{request_item.get('segment', 'N/A')} / "
                        f"{request_item.get('risk', 'N/A')}"
                    )

                    request_id = request_item.get("id")
                    if row[4].button(
                        "Ver",
                        key=f"retention_request_details_{request_id}",
                        help=f"Show details for request #{request_id}"
                    ):
                        current_id = st.session_state["selected_retention_request_id"]
                        st.session_state["selected_retention_request_id"] = (
                            None if current_id == request_id else request_id
                        )

            previous_col, page_col, next_col = st.columns([1, 2, 1])

            with previous_col:
                if st.button(
                    "Previous",
                    key="retention_requests_previous_page",
                    disabled=current_page == 1
                ):
                    st.session_state["retention_requests_page"] = current_page - 1
                    st.session_state["selected_retention_request_id"] = None
                    st.rerun()

            with page_col:
                st.markdown(
                    f"<div style='text-align: center; padding-top: 0.4rem;'>"
                    f"Page {current_page} of {total_pages}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with next_col:
                if st.button(
                    "Next",
                    key="retention_requests_next_page",
                    disabled=current_page == total_pages
                ):
                    st.session_state["retention_requests_page"] = current_page + 1
                    st.session_state["selected_retention_request_id"] = None
                    st.rerun()

            selected_request = next(
                (
                    request_item
                    for request_item in page_requests
                    if request_item.get("id")
                    == st.session_state["selected_retention_request_id"]
                ),
                None
            )

            if selected_request:
                show_retention_request_details(selected_request)


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