from fpdf import FPDF
import pandas as pd

def get_prob_color(prob):
    if prob < 0.3:
        return (255, 255, 153), 'black'   # Light Yellow
    elif prob < 0.7:
        return (255, 204, 102), 'black'   # Orange
    else:
        return (255, 102, 102), 'black'   # Red

def generate_pdf(df, highlight_columns=None):
    if highlight_columns is None:
        highlight_columns = []

    # ✅ Always-include core columns
    important_cols = ["TransactionID", "TransactionAmount", "TransactionDate", "FraudProbability"]

    # ✅ Add filtered columns to the export list (if not already there)
    export_cols = important_cols.copy()
    for col in highlight_columns:
        if col not in export_cols and col in df.columns:
            export_cols.append(col)

    # ✅ Create new export DataFrame
    export_df = df[export_cols].copy()

    # ✅ Setup PDF
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    # Title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Fraud Detection Report", ln=1, align='C')
    pdf.ln(4)
    pdf.set_font("Arial", size=10)

    # Layout sizing
    num_cols = len(export_cols)
    max_width = 277
    col_width = max_width / num_cols
    col_widths = [col_width] * num_cols

    # ✅ Header row with highlight
    for i, col in enumerate(export_cols):
        if col in highlight_columns:
            pdf.set_fill_color(255, 255, 153)  # yellow
        else:
            pdf.set_fill_color(230, 230, 230)  # gray
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[i], 10, col, border=1, fill=True)
    pdf.ln()

    # ✅ Row data
    for _, row in export_df.iterrows():
        for i, col in enumerate(export_cols):
            val = row[col]

            # Format values
            if isinstance(val, float) and 'Amount' in col:
                val_str = f"Rs.{val:,.2f}"
            elif isinstance(val, float) and 'Probability' in col:
                val_str = f"{val:.2%}"
            elif isinstance(val, pd.Timestamp):
                val_str = val.strftime('%Y-%m-%d')
            else:
                val_str = str(val)

            # Style conditions
            if col == 'FraudProbability':
                bg_color, txt_color = get_prob_color(row[col])
                pdf.set_fill_color(*bg_color)
                pdf.set_text_color(0, 0, 0 if txt_color == 'black' else 255)
                pdf.cell(col_widths[i], 10, val_str, border=1, fill=True)

            elif col in highlight_columns:
                pdf.set_fill_color(255, 255, 153)  # yellow
                pdf.set_text_color(0, 0, 0)
                pdf.cell(col_widths[i], 10, val_str, border=1, fill=True)

            else:
                pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(col_widths[i], 10, val_str, border=1, fill=True)

        pdf.ln()

    # ✅ Save PDF
    pdf.output("fraud_report.pdf")
