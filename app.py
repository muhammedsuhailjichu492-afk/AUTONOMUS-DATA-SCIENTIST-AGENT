

import os
import shutil
import tempfile
import zipfile
import io
import base64

import pandas as pd
import streamlit as st

from agent_core.orchestrator import AutonomousDataScientistAgent

LLM_MODEL_OPTIONS = [
    "claude-sonnet-4-6",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
]

st.set_page_config(
    page_title="Autonomous Data Scientist Agent",
    page_icon="🤖",
    layout="wide",
)


def _set_background(image_path: str) -> None:
    """Sets a fixed page background image with a light overlay so text and
    cards on top stay fully readable."""
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(rgba(255, 255, 255, 0.90), rgba(255, 255, 255, 0.90)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        [data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_set_background(os.path.join(os.path.dirname(__file__), "assets", "hero_background.png"))

# ----------------------------------------------------------------------
# Custom styling — clean, professional look on top of the light theme
# set in .streamlit/config.toml
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Page title block */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: #111827 !important;
        padding-bottom: 0.1rem;
    }
    h2, h3 {
        color: #1F2937 !important;
        font-weight: 600 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F4F6F9;
        border-right: 1px solid #E5E7EB;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.15rem !important;
    }

    /* Primary action button */
    .stButton > button[kind="primary"] {
        background-color: #2563EB;
        border: none;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8;
    }

    /* Success / metric banner after a run */
    div[data-testid="stAlertContainer"] {
        border-radius: 10px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #4B5563;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563EB;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #2563EB !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #6B7280;
    }

    /* Dataframe / table container */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        overflow: hidden;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF;
        border: 1.5px dashed #C7D2FE;
        border-radius: 10px;
    }

    /* Status widget (run progress) */
    div[data-testid="stStatusWidget"] {
        border-radius: 10px;
        border: 1px solid #E5E7EB;
    }

    /* Images (plots) get a card frame */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "run_dir" not in st.session_state:
    st.session_state.run_dir = None
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer) tuples

# ----------------------------------------------------------------------
# Sidebar: inputs
# ----------------------------------------------------------------------
st.sidebar.title("🤖 Autonomous Data Scientist Agent")
st.sidebar.caption("Profiling → Cleaning → EDA → Feature Engineering → Model Selection → Training → Evaluation → Report")

uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])

target_col = None
if uploaded_file is not None:
    try:
        preview_df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        st.session_state.df_preview = preview_df
        target_col = st.sidebar.selectbox("Target column to predict", options=list(preview_df.columns))
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

objective = st.sidebar.text_area(
    "Business objective (optional)",
    placeholder="e.g. Predict customer churn to prioritize retention outreach.",
)

with st.sidebar.expander("Advanced settings"):
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.slider("Cross-validation folds", 2, 10, 5, 1)

with st.sidebar.expander("🧠 LLM / Agent settings"):
    use_llm = st.checkbox(
        "Enable LLM-assisted decisions & narrative", value=True,
        help="Lets an LLM advise on cleaning/model-shortlist choices, write the "
             "report summary in plain language, and power the 'Ask the Agent' chat. "
             "Every LLM step has a rule-based fallback, so the pipeline still runs "
             "fully without a key — it just skips these extras.",
    )
    llm_api_key = st.text_input(
        "Anthropic API key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-...",
        help="Falls back to the ANTHROPIC_API_KEY environment variable if left blank. "
             "Never stored anywhere except this session.",
    )
    llm_model = st.selectbox("Model", options=LLM_MODEL_OPTIONS, index=0)

run_clicked = st.sidebar.button(
    "▶ Run Agent", type="primary", use_container_width=True,
    disabled=(uploaded_file is None or target_col is None),
)

st.sidebar.divider()
if st.session_state.result is not None:
    if st.sidebar.button("🔄 Reset", use_container_width=True):
        st.session_state.result = None
        st.session_state.run_dir = None
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title("Autonomous Data Scientist Agent")
st.caption("Upload a dataset and let the agent handle profiling, cleaning, EDA, modeling, and reporting end to end.")

if uploaded_file is None:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    st.markdown(
        """
        This agent autonomously performs a full data-science workflow on any
        tabular dataset you upload:

        1. **Data Profiling** — inspects rows, columns, types, missing data
        2. **Cleaning** — drops junk columns, imputes missing values, caps outliers
        3. **EDA** — generates distribution, correlation, and target plots
        4. **Feature Engineering** — scales numeric features, encodes categoricals
        5. **Model Selection** — shortlists candidate models for your task
        6. **Training** — cross-validates every candidate
        7. **Evaluation** — scores the best model on a held-out test set
        8. **Report** — writes everything up into a downloadable Markdown report
        """
    )
    st.stop()

if st.session_state.df_preview is not None and st.session_state.result is None:
    st.subheader("Dataset preview")
    st.dataframe(st.session_state.df_preview.head(20), use_container_width=True)
    st.caption(f"{st.session_state.df_preview.shape[0]} rows × {st.session_state.df_preview.shape[1]} columns")

# ----------------------------------------------------------------------
# Run the agent
# ----------------------------------------------------------------------
if run_clicked:
    run_dir = tempfile.mkdtemp(prefix="agent_run_")
    csv_path = os.path.join(run_dir, "input.csv")
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    output_dir = os.path.join(run_dir, "outputs")

    progress = st.empty()
    stages = [
        "Loading dataset", "Data Profiling", "Data Cleaning", "Exploratory Data Analysis",
        "Feature Engineering", "Model Selection", "Training & Cross-Validation",
        "Evaluation", "Report Generation",
    ]
    with st.status("Running the agent...", expanded=True) as status:
        for s in stages[:1]:
            st.write(f"• {s}")
        try:
            agent = AutonomousDataScientistAgent(
                output_dir=output_dir, test_size=test_size, cv_folds=cv_folds,
                use_llm=use_llm, api_key=llm_api_key or None, llm_model=llm_model,
            )
            for s in stages[1:]:
                st.write(f"• {s}")
            result = agent.run(csv_path=csv_path, target_col=target_col, business_objective=objective)
            status.update(label="Run complete ✅", state="complete", expanded=False)
        except ValueError as e:
            status.update(label="Run failed ❌", state="error")
            st.error(str(e))
            shutil.rmtree(run_dir, ignore_errors=True)
            st.stop()
        except Exception as e:
            status.update(label="Run failed ❌", state="error")
            st.exception(e)
            shutil.rmtree(run_dir, ignore_errors=True)
            st.stop()

    st.session_state.result = result
    st.session_state.run_dir = run_dir
    st.session_state.chat_history = []

# ----------------------------------------------------------------------
# Display results
# ----------------------------------------------------------------------
result = st.session_state.result
if result is not None:
    st.success(f"Best model: **{result.best_model_name}**")

    tabs = st.tabs(
        ["Summary", "Profiling", "EDA", "Leaderboard", "Evaluation", "Report / Download", "🤖 Ask the Agent"]
    )

    with tabs[0]:
        if result.llm_enabled:
            st.caption(f"🧠 LLM-assisted run — {result.llm_status}")
        else:
            st.caption(f"⚙️ Rule-based run only — {result.llm_status}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", result.profile.n_rows)
        c2.metric("Columns (after cleaning)", result.profile.n_cols)
        c3.metric("Task type", result.profile.task_type.capitalize())
        st.markdown("### Test-set metrics")
        st.table(pd.DataFrame(result.metrics.items(), columns=["Metric", "Value"]).set_index("Metric"))
        report_path = result.report_path
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        try:
            summary_section = report_text.split("## Summary")[1]
            st.markdown("### Summary")
            st.markdown(summary_section)
        except IndexError:
            pass
        if result.cleaning_advice or result.model_advice:
            with st.expander("🤖 Agent decisions (cleaning & model shortlist rationale)"):
                if result.cleaning_advice:
                    st.markdown(f"**Cleaning strategy** ({result.cleaning_advice.source}): {result.cleaning_advice.rationale}")
                if result.model_advice:
                    st.markdown(f"**Model shortlist** ({result.model_advice.source}): {result.model_advice.rationale}")

    with tabs[1]:
        st.markdown(result.profile.to_markdown())
        st.markdown("#### Cleaning actions")
        agent_ref = None  # cleaning log lives on the agent's cleaner; re-derive from report
        report_path = result.report_path
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        # extract the cleaning section for direct display
        try:
            section = report_text.split("## Data Cleaning Actions")[1].split("---")[0]
            st.markdown(section)
        except IndexError:
            pass

    with tabs[2]:
        plot_dir = st.session_state.run_dir and os.path.join(st.session_state.run_dir, "outputs")
        eda_files = [
            "correlation_heatmap.png", "distributions.png", "boxplots_vs_target.png",
            "target_distribution.png", "categorical_counts.png",
        ]
        cols = st.columns(2)
        i = 0
        for fname in eda_files:
            fpath = os.path.join(plot_dir, fname)
            if os.path.exists(fpath):
                cols[i % 2].image(fpath, caption=fname, use_container_width=True)
                i += 1

    with tabs[3]:
        try:
            section = report_text.split("## Model Selection & Training Leaderboard")[1].split("**Selected model:**")[0]
            st.markdown(section)
        except IndexError:
            pass
        st.markdown(f"**Selected model:** `{result.best_model_name}`")

    with tabs[4]:
        st.markdown("### Test-set evaluation")
        st.table(pd.DataFrame(result.metrics.items(), columns=["Metric", "Value"]).set_index("Metric"))
        eval_plot_name = "confusion_matrix.png" if result.profile.task_type == "classification" else "residuals.png"
        eval_plot_path = os.path.join(plot_dir, eval_plot_name)
        if os.path.exists(eval_plot_path):
            st.image(eval_plot_path, caption=eval_plot_name, use_container_width=False, width=500)

    with tabs[5]:
        st.markdown("### Full report")
        st.markdown(report_text)
        st.download_button(
            "⬇ Download report.md", data=report_text,
            file_name="report.md", mime="text/markdown",
        )

        # zip up the whole outputs folder (report + all plots) for one-click download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(plot_dir):
                zf.write(os.path.join(plot_dir, fname), arcname=fname)
        st.download_button(
            "⬇ Download all outputs (.zip)", data=zip_buffer.getvalue(),
            file_name="agent_outputs.zip", mime="application/zip",
        )

    with tabs[6]:
        st.markdown("### Ask the Agent")
        st.caption(
            "Ask follow-up questions about this run — why a column was dropped, "
            "what a metric means, which model to pick under different priorities, etc. "
            "Answers are grounded in this run's profile, logs, leaderboard, and metrics."
        )
        if not result.llm_enabled:
            st.info(
                "Chat requires an Anthropic API key. Add one under "
                "'🧠 LLM / Agent settings' in the sidebar and re-run the agent."
            )
        else:
            for q, a in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    st.markdown(a)

            question = st.chat_input("Ask a question about this run...")
            if question:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = result.chat_agent.ask(question, st.session_state.chat_history)
                    st.markdown(answer)
                st.session_state.chat_history.append((question, answer))