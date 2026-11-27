import os
import joblib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from src.config import config
from src.data_loader import fetch_ames_housing_data, preprocess_data


def _eval_split(y_true: pd.Series, preds: np.ndarray, median_price: float) -> dict:
    """Compute regression and stratified segment metrics on a holdout split."""
    residuals = y_true - preds
    rmse = float(root_mean_squared_error(y_true, preds))
    std_mask = y_true < 400000
    lux_mask = y_true >= 400000
    std_rmse = float(root_mean_squared_error(y_true[std_mask], preds[std_mask]))
    lux_rmse = float(root_mean_squared_error(y_true[lux_mask], preds[lux_mask]))
    return {
        "rmse": rmse,
        "rel_rmse_pct": (rmse / median_price) * 100.0,
        "mae": float(mean_absolute_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, preds) * 100.0),
        "res_mean": float(residuals.mean()),
        "res_std": float(residuals.std()),
        "max_abs_err": float(np.abs(residuals).max()),
        "std_rmse": std_rmse,
        "lux_rmse": lux_rmse,
        "lux_ratio": lux_rmse / std_rmse,
    }


def compute_report_metrics() -> dict:
    """Load data, test split, and trained models to compute empirical report metrics."""
    df = fetch_ames_housing_data()
    processed = preprocess_data(df)
    X_test = pd.read_csv(os.path.join(config.paths.data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(config.paths.data_dir, "y_test.csv")).squeeze()

    lr = joblib.load(os.path.join(config.paths.model_dir, "linear_regression.joblib"))
    rf = joblib.load(os.path.join(config.paths.model_dir, "random_forest.joblib"))

    lr_preds = lr.predict(X_test)
    rf_preds = rf.predict(X_test)

    target = df["Sale_Price"]
    median_price = float(target.median())
    na_counts = df.isna().sum()
    missing_cols = na_counts[na_counts > 0].sort_values(ascending=False).to_dict()

    numeric_df = df.select_dtypes(include=["number"])
    corrs = numeric_df.corr()["Sale_Price"].sort_values(ascending=False).iloc[1:6]
    top_corrs = [(col, float(val)) for col, val in corrs.items()]
    garage_corr = float(df["Garage_Cars"].corr(df["Garage_Area"]))

    imp_pairs = sorted(zip(X_test.columns, rf.feature_importances_), key=lambda x: -x[1])
    top_rf_imp = [(col, float(val)) for col, val in imp_pairs[:6]]
    overall_qual_total_imp = float(
        sum(val for col, val in imp_pairs if col.startswith("Overall_Qual_"))
    )

    return {
        "raw_rows": int(df.shape[0]),
        "raw_cols": int(df.shape[1]),
        "proc_features": int(processed.shape[1] - 1),
        "n_train": int(len(processed) - len(X_test)),
        "n_test": int(len(X_test)),
        "target_mean": float(target.mean()),
        "target_median": median_price,
        "target_std": float(target.std()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_skew": float(target.skew()),
        "missing_cols": {k: int(v) for k, v in missing_cols.items()},
        "n_standard_test": int((y_test < 400000).sum()),
        "n_luxury_test": int((y_test >= 400000).sum()),
        "n_luxury_all": int((target >= 400000).sum()),
        "lr": _eval_split(y_test, lr_preds, median_price),
        "rf": _eval_split(y_test, rf_preds, median_price),
        "top_corrs": top_corrs,
        "top_rf_imp": top_rf_imp,
        "overall_qual_total_imp": overall_qual_total_imp,
        "garage_corr": garage_corr,
    }


def _build_table(data: list[list[str]], col_widths: list[float]) -> Table:
    """Create a compact IEEE-styled booktabs table."""
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 9.5),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def generate_report():
    m = compute_report_metrics()
    lr = m["lr"]
    rf = m["rf"]
    report_path = os.path.join(config.paths.reports_dir, "Housing_Price_Prediction_Report.pdf")

    page_width, page_height = letter
    margin_left = 45
    margin_right = 45
    margin_top = 60
    margin_bottom = 60

    content_width = page_width - margin_left - margin_right
    gutter = 18
    column_width = (content_width - gutter) / 2.0

    doc = BaseDocTemplate(
        report_path,
        pagesize=letter,
        leftMargin=margin_left,
        rightMargin=margin_right,
        topMargin=margin_top,
        bottomMargin=margin_bottom,
    )

    title_height = 235
    col_height_first = page_height - margin_top - title_height - margin_bottom
    col_height_later = page_height - margin_top - margin_bottom

    first_page_template = PageTemplate(
        id="FirstPage",
        frames=[
            Frame(
                margin_left,
                page_height - margin_top - title_height,
                content_width,
                title_height,
                id="title",
                showBoundary=0,
            ),
            Frame(margin_left, margin_bottom, column_width, col_height_first, id="first_left", showBoundary=0),
            Frame(
                margin_left + column_width + gutter,
                margin_bottom,
                column_width,
                col_height_first,
                id="first_right",
                showBoundary=0,
            ),
        ],
    )
    later_page_template = PageTemplate(
        id="LaterPages",
        frames=[
            Frame(margin_left, margin_bottom, column_width, col_height_later, id="left", showBoundary=0),
            Frame(
                margin_left + column_width + gutter,
                margin_bottom,
                column_width,
                col_height_later,
                id="right",
                showBoundary=0,
            ),
        ],
    )
    doc.addPageTemplates([first_page_template, later_page_template])

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "IEEE_Title",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    style_author = ParagraphStyle(
        "IEEE_Author",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    style_abstract = ParagraphStyle(
        "IEEE_Abstract",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=9,
        leading=11,
        alignment=TA_JUSTIFY,
        leftIndent=18,
        rightIndent=18,
        spaceAfter=10,
    )
    style_heading = ParagraphStyle(
        "IEEE_Heading",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=10,
        leading=12,
        alignment=TA_CENTER,
        spaceBefore=10,
        spaceAfter=5,
        textTransform="uppercase",
        keepWithNext=True,
    )
    style_body = ParagraphStyle(
        "IEEE_Body",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9.5,
        leading=11.5,
        alignment=TA_JUSTIFY,
        firstLineIndent=10,
        spaceAfter=2,
    )
    style_caption = ParagraphStyle(
        "IEEE_Caption",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=8,
    )

    c0_name, c0_val = m["top_corrs"][0]
    c1_name, c1_val = m["top_corrs"][1]
    c2_name, c2_val = m["top_corrs"][2]
    c3_name, c3_val = m["top_corrs"][3]
    c4_name, c4_val = m["top_corrs"][4]

    f0_name, f0_val = m["top_rf_imp"][0]
    f1_name, f1_val = m["top_rf_imp"][1]
    f2_name, f2_val = m["top_rf_imp"][2]
    f3_name, f3_val = m["top_rf_imp"][3]
    f4_name, f4_val = m["top_rf_imp"][4]
    f5_name, f5_val = m["top_rf_imp"][5]

    story = []

    # --- Title & Abstract ---
    story.append(Paragraph("Ames Housing Price Prediction: Empirical Evaluation and Error Analysis", style_title))
    story.append(Paragraph("GaneshArwan<br/>github.com/GaneshArwan/housing_price_prediction", style_author))
    abstract_text = (
        f"<b><i>Abstract</i>—This study evaluates an end-to-end regression pipeline on the Ames Housing Dataset "
        f"({m['raw_rows']:,} residential sales, {m['raw_cols']} raw columns, {m['proc_features']} encoded features). "
        f"We compare Multiple Linear Regression against a 100-tree Random Forest Regressor on a held-out 20% test split "
        f"(n={m['n_test']}). Random Forest achieves an RMSE of ${rf['rmse']:,.2f} (R<super>2</super>={rf['r2']:.4f}, "
        f"MAE=${rf['mae']:,.2f}) compared to ${lr['rmse']:,.2f} (R<super>2</super>={lr['r2']:.4f}, "
        f"MAE=${lr['mae']:,.2f}) for Linear Regression. Above-ground living area ({c0_name}, r={c0_val:.2f}) and "
        f"garage capacity ({f0_name}, {f0_val * 100:.1f}% MDI) dominate linear and tree splits, while luxury homes "
        f"(&ge;$400,000) exhibit {rf['lux_ratio']:.1f}x higher RMSE than standard homes.</b>"
    )
    story.append(Paragraph(abstract_text, style_abstract))

    story.append(FrameBreak())
    story.append(NextPageTemplate("LaterPages"))

    # --- I. Introduction ---
    story.append(Paragraph("I. Introduction", style_heading))
    story.append(
        Paragraph(
            f"Residential property valuation requires modeling structural attributes, neighborhood effects, and "
            f"non-linear feature interactions. The Ames Housing Dataset (De Cock, 2011) records {m['raw_rows']:,} "
            f"home sales in Ames, Iowa (2006–2010) across {m['raw_cols']} columns (80 explanatory variables plus "
            f"<i>Sale_Price</i>).",
            style_body,
        )
    )
    story.append(
        Paragraph(
            f"Target sale prices span ${m['target_min']:,.0f} to ${m['target_max']:,.0f}, with a right-skewed "
            f"distribution (skewness={m['target_skew']:.2f}): the mean is ${m['target_mean']:,.0f}, the median is "
            f"${m['target_median']:,.0f}, and the standard deviation is ${m['target_std']:,.0f}. Only "
            f"{m['n_luxury_all']} transactions ({m['n_luxury_all'] / m['raw_rows'] * 100:.1f}%) equal or exceed "
            f"$400,000, creating tail sparsity that affects squared-error loss.",
            style_body,
        )
    )

    # --- II. Methodology ---
    story.append(Paragraph("II. Methodology", style_heading))
    miss_desc = ", ".join(f"<i>{k}</i> ({v:,})" for k, v in m["missing_cols"].items())
    story.append(
        Paragraph(
            f"<b>Preprocessing:</b> Raw ingestion drops identifier and sparse legacy columns if present "
            f"(<i>Id</i>, <i>PID</i>, <i>Alley</i>, <i>PoolQC</i>, <i>Fence</i>, <i>MiscFeature</i>) and renames "
            f"<i>Sale_Price</i> to <i>SalePrice</i>. In this mirror, missing values occur in two columns: {miss_desc}. "
            f"Numeric missing values are imputed with column medians, categorical missing values are filled with "
            f"<i>'Unknown'</i>, and categorical features are one-hot encoded with <i>drop_first=True</i>, producing "
            f"{m['proc_features']} model features without feature scaling.",
            style_body,
        )
    )
    story.append(
        Paragraph(
            f"<b>Training and Inference Parity:</b> Data is partitioned 80/20 (<i>random_state={config.model.random_state}</i>) "
            f"into {m['n_train']:,} training and {m['n_test']:,} test observations. We fit a baseline ordinary least "
            f"squares <i>LinearRegression</i> model and a <i>RandomForestRegressor</i> "
            f"(<i>n_estimators={config.model.rf_n_estimators}</i>, <i>max_depth={config.model.rf_max_depth}</i>, "
            f"<i>random_state={config.model.random_state}</i>), tracking parameters, metrics, and artifacts in MLflow. "
            f"Single-row Streamlit inference sets <i>drop_first=False</i> and reindexes onto the {m['proc_features']} "
            f"training columns so selected categories are preserved. Predictions pass through Pydantic guardrails "
            f"enforcing finite floats within $1,000 to $10,000,000.",
            style_body,
        )
    )

    # --- III. Experimental Results & Error Analysis ---
    story.append(Paragraph("III. Experimental Results &amp; Error Analysis", style_heading))
    story.append(
        Paragraph(
            f"Table I summarizes holdout performance across all {m['n_test']} test homes. Random Forest reduces RMSE "
            f"by ${lr['rmse'] - rf['rmse']:,.2f} ({(1 - rf['rmse'] / lr['rmse']) * 100:.2f}%), raises R<super>2</super> "
            f"from {lr['r2']:.4f} to {rf['r2']:.4f}, and lowers maximum absolute error from "
            f"${lr['max_abs_err']:,.0f} to ${rf['max_abs_err']:,.0f}. Conversely, Linear Regression achieves a slightly "
            f"lower MAE (${lr['mae']:,.2f} vs. ${rf['mae']:,.2f}) and MAPE ({lr['mape_pct']:.2f}% vs. "
            f"{rf['mape_pct']:.2f}%), indicating tighter typical-home error offset by larger tail outliers.",
            style_body,
        )
    )

    table1_data = [
        [f"Metric (Test n={m['n_test']})", "Linear Reg.", "Random Forest"],
        ["RMSE ($)", f"${lr['rmse']:,.2f}", f"${rf['rmse']:,.2f}"],
        ["Relative RMSE (% Med.)", f"{lr['rel_rmse_pct']:.2f}%", f"{rf['rel_rmse_pct']:.2f}%"],
        ["MAE ($)", f"${lr['mae']:,.2f}", f"${rf['mae']:,.2f}"],
        ["R-Squared (R2)", f"{lr['r2']:.4f}", f"{rf['r2']:.4f}"],
        ["MAPE (%)", f"{lr['mape_pct']:.2f}%", f"{rf['mape_pct']:.2f}%"],
        ["Mean Residual ($)", f"${lr['res_mean']:,.2f}", f"${rf['res_mean']:,.2f}"],
        ["Max |Error| ($)", f"${lr['max_abs_err']:,.0f}", f"${rf['max_abs_err']:,.0f}"],
    ]
    story.append(
        KeepTogether(
            [
                Spacer(1, 4),
                Paragraph("TABLE I: HOLDOUT EVALUATION METRICS", style_caption),
                _build_table(table1_data, [112, 69, 69]),
                Spacer(1, 6),
            ]
        )
    )

    table2_data = [
        ["Segment", "n", "LR RMSE", "RF RMSE"],
        ["Standard (<$400k)", str(m["n_standard_test"]), f"${lr['std_rmse']:,.0f}", f"${rf['std_rmse']:,.0f}"],
        ["Luxury (>=$400k)", str(m["n_luxury_test"]), f"${lr['lux_rmse']:,.0f}", f"${rf['lux_rmse']:,.0f}"],
        ["Luxury / Std Ratio", "—", f"{lr['lux_ratio']:.2f}x", f"{rf['lux_ratio']:.2f}x"],
    ]
    story.append(
        KeepTogether(
            [
                Spacer(1, 2),
                Paragraph("TABLE II: STRATIFIED SEGMENT ERROR", style_caption),
                _build_table(table2_data, [104, 28, 59, 59]),
                Spacer(1, 6),
            ]
        )
    )

    story.append(
        Paragraph(
            f"Table II quantifies error across price regimes. On standard homes (&lt;$400,000, n={m['n_standard_test']}), "
            f"Random Forest RMSE is ${rf['std_rmse']:,.0f}. On luxury homes (&ge;$400,000, n={m['n_luxury_test']}), "
            f"RMSE rises to ${rf['lux_rmse']:,.0f} ({rf['lux_ratio']:.2f}x higher), confirming sparse luxury training "
            f"coverage rather than uniform variance.",
            style_body,
        )
    )

    def add_figure(filename: str, caption: str, width: float = column_width):
        filepath = os.path.join(config.paths.reports_dir, filename)
        if os.path.exists(filepath):
            iw, ih = ImageReader(filepath).getSize()
            height = width * (ih / float(iw))
            img = Image(filepath, width=width, height=height)
            cap = Paragraph(caption, style_caption)
            story.append(KeepTogether([Spacer(1, 4), img, cap, Spacer(1, 4)]))
        else:
            story.append(Paragraph(f"[Missing Figure: {filename}]", style_body))

    add_figure(
        "correlation_heatmap.png",
        "Fig. 1. Pearson correlation heatmap for the top 10 numeric features against Sale_Price.",
    )
    story.append(
        Paragraph(
            f"In Fig. 1, above-ground living area (<i>{c0_name}</i>, r={c0_val:.4f}) shows the strongest linear "
            f"association with sale price, followed by garage capacity (<i>{c1_name}</i>, r={c1_val:.4f}), garage area "
            f"(<i>{c2_name}</i>, r={c2_val:.4f}), basement area (<i>{c3_name}</i>, r={c3_val:.4f}), and first-floor "
            f"area (<i>{c4_name}</i>, r={c4_val:.4f}). Because <i>Overall_Qual</i> is stored as categorical labels "
            f"in this dataset, it is excluded from numeric Pearson correlations.",
            style_body,
        )
    )

    add_figure(
        "feature_importance.png",
        "Fig. 2. Top 20 Random Forest Mean Decrease in Impurity (MDI) feature importances.",
    )
    story.append(
        Paragraph(
            f"Fig. 2 ranks Random Forest MDI splits: <i>{f0_name}</i> ({f0_val * 100:.2f}%), <i>{f1_name}</i> "
            f"({f1_val * 100:.2f}%), <i>{f2_name}</i> ({f2_val * 100:.2f}%), <i>{f3_name}</i> ({f3_val * 100:.2f}%), "
            f"<i>{f4_name}</i> ({f4_val * 100:.2f}%), and <i>{f5_name}</i> ({f5_val * 100:.2f}%). Notably, "
            f"one-hot encoding dilutes <i>Overall_Qual</i> across nine separate dummy columns whose combined MDI "
            f"importance is {m['overall_qual_total_imp'] * 100:.2f}%, masking ordinal rank information.",
            style_body,
        )
    )

    add_figure(
        "residual_plot.png",
        f"Fig. 3. Random Forest residuals (Actual minus Predicted SalePrice) on the test split (n={m['n_test']}).",
    )
    story.append(
        Paragraph(
            f"Fig. 3 shows near-zero mean bias (${rf['res_mean']:,.2f}; residual SD=${rf['res_std']:,.2f}) and "
            f"compact errors below $350,000, but widening spread above $400,000. Tree averaging caps extreme upper-tail "
            f"predictions when luxury comparables are scarce in training leaves.",
            style_body,
        )
    )

    # --- IV. Limitations & Recommendations ---
    story.append(Paragraph("IV. Limitations &amp; Recommendations", style_heading))
    story.append(
        Paragraph(
            "Four concrete improvements follow from this audit: (1) map ordinal quality and condition columns "
            "(<i>Overall_Qual</i>, <i>Overall_Cond</i>, <i>Exter_Qual</i>, <i>Kitchen_Qual</i>) to ordered integers "
            "(1–10 / 1–5) instead of sparse one-hot dummies; (2) train against log-transformed <i>log1p(SalePrice)</i> "
            "to stabilize variance on right-skewed luxury sales; (3) benchmark regularized gradient boosting "
            f"(LightGBM / XGBoost) with 5-fold cross-validation; and (4) mitigate collinearity between "
            f"<i>Garage_Cars</i> and <i>Garage_Area</i> (r={m['garage_corr']:.2f}) when interpreting feature attributions.",
            style_body,
        )
    )

    # --- V. References ---
    story.append(Paragraph("V. References", style_heading))
    story.append(
        Paragraph(
            '[1] D. De Cock, "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression '
            'Project," <i>J. Statistics Education</i>, vol. 19, no. 3, 2011.',
            style_body,
        )
    )
    story.append(
        Paragraph(
            '[2] L. Breiman, "Random Forests," <i>Machine Learning</i>, vol. 45, no. 1, pp. 5–32, 2001.',
            style_body,
        )
    )
    story.append(
        Paragraph(
            '[3] M. Zaharia et al., "Accelerating the Machine Learning Lifecycle with MLflow," <i>IEEE Data Eng. '
            "Bull.</i>, vol. 41, no. 4, pp. 39–45, 2018.",
            style_body,
        )
    )

    doc.build(story)
    print(f"IEEE formatted PDF report generated at {report_path}")


if __name__ == "__main__":
    generate_report()
