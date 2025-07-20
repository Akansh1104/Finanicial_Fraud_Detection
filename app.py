import streamlit as st
from model import FraudDetector
import pandas as pd
from report_generator import generate_pdf
import math
import shap
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="💳")
st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        .block-container {padding: 2rem 2rem;}
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            font-size: 16px;
            border-radius: 10px;
            transition-duration: 0.4s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            color: white;
        }
        .stDownloadButton>button {
            background-color: #ff7f50;
            color: white;
            border-radius: 10px;
        }
        h1, h2, h3, h4, h5 {
            color: #2c3e50;
        }
        .metric-label {
            color: #34495e;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Financial Fraud Detection Dashboard")

uploaded = st.file_uploader("📄 Upload CSV File", type=["csv"])

if uploaded:
    with open("data/dataset.csv", "wb") as f:
        f.write(uploaded.getbuffer())

    detector = FraudDetector("data/dataset.csv")
    result = detector.predict()
    frauds = result[result['IsFraud'] == 1]

    st.success(f"✅ Found {len(frauds)} potential frauds out of {len(result)} transactions.")



    with st.expander("📊 View All Transactions", expanded=False):
        st.dataframe(result.style.set_properties(**{'background-color': '#fefefe'}), use_container_width=True)

    if not frauds.empty:
        

        types = frauds['TransactionType'].unique().tolist()
        types.insert(0, "All")
        locs = frauds['Location'].unique().tolist()
        locs.insert(0, "All")
        channels = frauds['Channel'].unique().tolist()
        channels.insert(0, "All")

        min_amt = int(frauds['TransactionAmount'].min())
        max_amt = int(frauds['TransactionAmount'].max())
        frauds['TransactionDate'] = pd.to_datetime(frauds['TransactionDate'])
        min_date = frauds['TransactionDate'].min().date()
        max_date = frauds['TransactionDate'].max().date()

        # --- Filter UI ---
        st.markdown("### 🎛️ Filter Options")
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                type_sel = st.selectbox("Transaction Type", types, index=0)
            with col2:
                loc_sel = st.multiselect("Location(s)", locs, default=["All"])
            with col3:
                channel_sel = st.multiselect("Channel(s)", channels, default=["All"])
            with col4:
                amt_range = st.slider("Amount Range (₹)", min_amt, max_amt, (min_amt, max_amt))

            col5, col6 = st.columns(2)
            with col5:
                start_date = st.date_input("From", min_date)
            with col6:
                end_date = st.date_input("To", max_date)

        # --- Check for active filters ---
        filters_applied = (
            type_sel != "All" or
            loc_sel != ["All"] or
            channel_sel != ["All"] or
            amt_range != (min_amt, max_amt) or
            start_date != min_date or
            end_date != max_date
        )

        if filters_applied:
            if st.button("🔄 Reset Filters"):
                st.rerun()

        # --- Apply filters ---
        filtered = frauds.copy()
        if type_sel != "All":
            filtered = filtered[filtered['TransactionType'] == type_sel]
        if "All" not in loc_sel:
            filtered = filtered[filtered['Location'].isin(loc_sel)]
        if "All" not in channel_sel:
            filtered = filtered[filtered['Channel'].isin(channel_sel)]

        filtered = filtered[
            (filtered['TransactionAmount'].between(*amt_range)) &
            (filtered['TransactionDate'].dt.date.between(start_date, end_date))
        ]


        st.markdown(f"### 📋 {len(filtered)+1} Fraudulent Transactions Found")
        filtered['RiskTier'] = pd.cut(filtered['FraudProbability'], bins=[0, 0.3, 0.7, 1], labels=["Low", "Medium", "High"])
        st.dataframe(filtered.drop(columns=['FraudExplanation']), use_container_width=True)

        st.markdown("---")
        generate_pdf(filtered)
        if not filtered.empty and st.button("📄 Generate PDF Report"):
            with open("fraud_report.pdf", "rb") as f:
                st.download_button("Download PDF", f, file_name="fraud_report.pdf", mime="application/pdf")

        st.markdown("---")
        st.subheader("🧠 Fraud Explanation Explorer")

        fraud_labels = filtered.apply(
            lambda row: f"{row['TransactionID']} — ₹{row['TransactionAmount']:.2f} on {row['TransactionDate'].date()}", axis=1
        ).tolist()
        tx_id_map = dict(zip(fraud_labels, filtered.index))
        selected_label = st.selectbox("🔎 Select a Fraudulent Transaction:", fraud_labels)
        selected_idx = tx_id_map[selected_label]
        row = frauds.loc[selected_idx]

        st.markdown("#### 📌 Transaction Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Amount", f"₹{row['TransactionAmount']:.2f}")
            st.metric("Channel", row['Channel'])
            st.metric("Type", row['TransactionType'])
        with col2:
            st.metric("Balance", f"₹{row['AccountBalance']:.2f}")
            st.metric("Location", row['Location'])
            st.metric("Login Attempts", int(row['LoginAttempts']))

        prob = row['FraudProbability']
        summary_text = f"This transaction has a **{prob:.1%} fraud risk** due to:"
        st.markdown(f"#### 🛑 Risk Summary\n{summary_text}")

        explanation = row.get('FraudExplanation', 'Explanation not available.')
        st.markdown(f"""
        <div style='border: 1px solid #ccc; padding: 1rem; border-radius: 8px; background-color: #f9f9f9;'>
        <pre style='white-space: pre-wrap; font-size: 0.95rem;'>{explanation}</pre>
        </div>
        """, unsafe_allow_html=True)

        X_scaled = detector.preprocess()
        X_df = pd.DataFrame(X_scaled, columns=detector.feature_cols)
        explainer = shap.Explainer(detector.model, X_df)
        shap_values = explainer(X_df)
        
        with st.expander("🔬 View SHAP Waterfall Explanation"):
            st.info("This SHAP waterfall plot explains how each feature influenced the fraud risk prediction for this transaction.")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[selected_idx], max_display=10, show=False)
            st.pyplot(fig)
        
        with st.expander("📈 Fraud Trends Overview"):
            st.subheader("📈 Fraud Trends Overview")
            result['TransactionDate'] = pd.to_datetime(result['TransactionDate'])
            frauds['TransactionDate'] = pd.to_datetime(frauds['TransactionDate'])
            timeline = frauds.groupby(frauds['TransactionDate'].dt.date).size().reset_index(name='Frauds')
            fig_timeline = px.line(timeline, x='TransactionDate', y='Frauds', title='📊 Daily Fraud Trend', markers=True, line_shape='spline')
            fig_timeline.update_layout(template="plotly_white")
            st.plotly_chart(fig_timeline, use_container_width=True)

        if 'Location' in frauds.columns:
            with st.expander("🔬 Fraud Heatmap by Location"):
                st.subheader("🗺️ Fraud Heatmap by Location")
                loc_counts = frauds['Location'].value_counts().reset_index()
                loc_counts.columns = ['Location', 'Count']
                fig_map = px.bar(loc_counts, x='Location', y='Count', color='Count', title='🔴 Fraud Count by Location')
                fig_map.update_layout(template="plotly_white")
                st.plotly_chart(fig_map, use_container_width=True)

       

        with st.expander("📊 SHAP Feature Impact Summary"):
            st.markdown("This chart shows the most influential features in fraud predictions overall.")
            fig_bar, ax = plt.subplots(figsize=(10, 6))
            shap.plots.bar(shap_values, max_display=10, show=False)
            st.pyplot(fig_bar)

        st.markdown("---")
        st.subheader("🧪 Try Your Own Transaction")
        with st.form("simulate_form"):
            col1, col2 = st.columns(2)
            with col1:
                amount = st.number_input("Amount", value=50000.0)
                attempts = st.number_input("Login Attempts", value=2)
            with col2:
                duration = st.number_input("Transaction Duration (sec)", value=60)
                balance = st.number_input("Account Balance", value=10000.0)
            submit = st.form_submit_button("Simulate")

        if submit:
            st.markdown(f"**🧾 Prediction**: With ₹{amount}, {attempts} login attempts, and ₹{balance} balance...")
            risk = "High" if amount > 75000 or attempts >= 3 else "Moderate" if amount > 30000 else "Low"
            st.metric("Estimated Fraud Risk Tier", risk)
            st.info("This is a rough simulation based on typical fraud patterns. The real model uses more features.")

    else:
        st.warning("🚫 No frauds detected in the uploaded dataset.")
