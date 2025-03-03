from flask import Flask, request, render_template, send_file, flash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from scipy.stats import shapiro, pearsonr, chi2_contingency
import os
import io
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables
session_df = None
dataset_type = None

# Descriptions for each step
STEP_DESCRIPTIONS = {
    1: "Loads the uploaded CSV file into a Pandas DataFrame, providing the foundation for all subsequent analyses by converting raw data into a structured format.",
    2: "Examines the dataset's structure, including feature names, data types, initial rows, descriptive statistics, and statistical tests to understand its composition and properties.",
    3: "Identifies missing values in each column and offers advanced options to handle them, ensuring data completeness for accurate analysis.",
    4: "Detects outliers in numeric columns using z-scores and provides a method to remove them, improving data quality by addressing anomalies.",
    5: "Scales numeric data to a standardized range using customizable normalization methods, preparing it for machine learning or further analysis.",
    6: "Selects the most relevant features using Mutual Information and Chi-Squared methods, with a visualization to highlight feature importance.",
    7: "Generates visualizations tailored to the dataset type (e.g., boxplots and histograms for numerical data, word clouds for text), offering a visual summary of data distributions."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global session_df, dataset_type
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return render_template('index.html')
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return render_template('index.html')
    
    dataset_type = request.form.get('dataset_type', 'Numerical')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)
        session_df = df.copy()
        eda_results = run_all_eda_steps(df, dataset_type)
        os.remove(file_path)
        flash('File uploaded successfully', 'success')
        return render_template('results.html', dataset_type=dataset_type, step_descriptions=STEP_DESCRIPTIONS, **eda_results)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        flash(f'Error processing file: {str(e)}', 'error')
        return render_template('index.html')

@app.route('/handle_missing', methods=['POST'])
def handle_missing():
    global session_df, dataset_type
    if session_df is None:
        flash('No dataset available', 'error')
        return render_template('index.html')
    
    method = request.form.get('method', 'drop')
    original_rows = len(session_df)
    if method == 'drop':
        session_df = session_df.dropna()
    elif method == 'mean':
        session_df = session_df.fillna(session_df.mean(numeric_only=True))
    elif method == 'median':
        session_df = session_df.fillna(session_df.median(numeric_only=True))
    elif method == 'mode':
        session_df = session_df.fillna(session_df.mode().iloc[0])
    new_rows = len(session_df)
    eda_results = run_all_eda_steps(session_df, dataset_type)
    flash(f'Missing values handled ({method}): {original_rows - new_rows} rows affected', 'success')
    return render_template('results.html', dataset_type=dataset_type, step_descriptions=STEP_DESCRIPTIONS, **eda_results)

@app.route('/remove_outliers', methods=['POST'])
def remove_outliers():
    global session_df, dataset_type
    if session_df is None:
        flash('No dataset available', 'error')
        return render_template('index.html')
    
    original_rows = len(session_df)
    numeric_cols = session_df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        z_scores = np.abs((session_df[col] - session_df[col].mean()) / session_df[col].std())
        session_df = session_df[z_scores <= 3]
    new_rows = len(session_df)
    eda_results = run_all_eda_steps(session_df, dataset_type)
    flash(f'Outliers removed: {original_rows - new_rows} rows affected', 'success')
    return render_template('results.html', dataset_type=dataset_type, step_descriptions=STEP_DESCRIPTIONS, **eda_results)

@app.route('/normalize', methods=['POST'])
def normalize():
    global session_df, dataset_type
    if session_df is None:
        flash('No dataset available', 'error')
        return render_template('index.html')
    
    method = request.form.get('method', 'minmax')
    numeric_cols = session_df.select_dtypes(include=np.number).columns
    if numeric_cols.any():
        col = numeric_cols[0]
        if method == 'minmax':
            session_df[f'{col}_normalized'] = (session_df[col] - session_df[col].min()) / (session_df[col].max() - session_df[col].min())
        elif method == 'zscore':
            session_df[f'{col}_normalized'] = (session_df[col] - session_df[col].mean()) / session_df[col].std()
        elif method == 'log':
            session_df[f'{col}_normalized'] = np.log1p(session_df[col].clip(lower=0))
        flash(f'Normalized {col} using {method}', 'success')
    eda_results = run_all_eda_steps(session_df, dataset_type)
    return render_template('results.html', dataset_type=dataset_type, step_descriptions=STEP_DESCRIPTIONS, **eda_results)

@app.route('/download_report', methods=['GET'])
def download_report():
    global session_df, dataset_type
    if session_df is None:
        return "No dataset available", 400

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=50, leftMargin=36, rightMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Header', fontSize=10, textColor=colors.white, alignment=1))
    styles.add(ParagraphStyle(name='SectionTitle', fontSize=14, textColor=colors.darkblue, spaceAfter=10))
    styles.add(ParagraphStyle(name='Description', fontSize=10, textColor=colors.black, spaceAfter=5))

    def add_header(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.darkblue)
        canvas.rect(0, letter[1] - 40, letter[0], 40, fill=1)
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(colors.white)
        canvas.drawCentredString(letter[0]/2, letter[1] - 25, f"ExploreX Report - {dataset_type} Dataset | {datetime.now().strftime('%Y-%m-%d')}")
        canvas.restoreState()

    story = []
    PAGE_WIDTH = 540
    MAX_ROWS_PER_PAGE = 20

    def split_table(data, max_rows):
        return [data[i:i + max_rows] for i in range(0, len(data), max_rows)]

    # Summary
    story.append(Paragraph("Data Summary", styles['SectionTitle']))
    story.append(Paragraph("Overview of the dataset's key characteristics.", styles['Description']))
    summary_data = [
        ["Rows", len(session_df)],
        ["Columns", len(session_df.columns)],
        ["Missing Values", session_df.isnull().sum().sum()],
        ["Numeric Columns", len(session_df.select_dtypes(include=np.number).columns)],
        ["Categorical Columns", len(session_df.select_dtypes(include='object').columns)]
    ]
    story.append(Table(summary_data, colWidths=[270, 270], style=[('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('FONTSIZE', (0,0), (-1,-1), 7), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 12))

    # Step 1: Load Data
    story.append(Paragraph("1. Load Data", styles['SectionTitle']))
    story.append(Paragraph(STEP_DESCRIPTIONS[1], styles['Description']))
    story.append(Paragraph(f"The dataset has {len(session_df.columns)} features and {len(session_df.index)} tuples.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Step 2: Data Exploration
    story.append(Paragraph("2. Data Exploration", styles['SectionTitle']))
    story.append(Paragraph(STEP_DESCRIPTIONS[2], styles['Description']))
    features_data = [["Feature", "Data Type"]] + [[str(f), str(session_df[f].dtype)] for f in session_df.columns]
    for chunk in split_table(features_data, MAX_ROWS_PER_PAGE):
        story.append(KeepTogether(Table(chunk, colWidths=[270, 270], style=[('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('FONTSIZE', (0,0), (-1,-1), 7), ('ALIGN', (0,0), (-1,-1), 'CENTER')])))
        story.append(Spacer(1, 6))
    story.append(Spacer(1, 12))
    story.append(Paragraph("First 10 Rows:", styles['Normal']))
    first_10_data = [session_df.columns.tolist()] + session_df.head(10).values.tolist()
    num_cols = len(session_df.columns)
    col_width = min(100, PAGE_WIDTH / num_cols) if num_cols > 0 else PAGE_WIDTH
    for chunk in split_table(first_10_data, MAX_ROWS_PER_PAGE):
        story.append(KeepTogether(Table(chunk, colWidths=[col_width]*num_cols, style=[('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('FONTSIZE', (0,0), (-1,-1), 7), ('ALIGN', (0,0), (-1,-1), 'CENTER')])))
        story.append(Spacer(1, 6))
    if dataset_type == 'Numerical':
        story.append(Spacer(1, 12))
        story.append(Paragraph("Descriptive Statistics:", styles['Normal']))
        stats_df = session_df.describe()
        stats_data = [[""] + stats_df.columns.tolist()] + [[idx] + stats_df.loc[idx].tolist() for idx in stats_df.index]
        num_stats_cols = len(stats_df.columns) + 1
        col_width = min(80, PAGE_WIDTH / num_stats_cols) if num_stats_cols > 0 else PAGE_WIDTH
        story.append(Table(stats_data, colWidths=[col_width]*num_stats_cols, style=[('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('FONTSIZE', (0,0), (-1,-1), 7), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Statistical Tests:", styles['Normal']))
        normality_data = [["Column", "Shapiro p-value", "Normal?"]] + [[col, *shapiro(session_df[col].dropna())[:2], "Yes" if shapiro(session_df[col].dropna())[1] > 0.05 else "No"] for col in session_df.select_dtypes(include=np.number).columns]
        story.append(Table(normality_data, colWidths=[180, 180, 180], style=[('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('FONTSIZE', (0,0), (-1,-1), 7), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        if len(session_df.select_dtypes(include=np.number).columns) > 1:
            numeric_df = session_df.select_dtypes(include=np.number).dropna()
            if len(numeric_df) >= 2:
                corr_data = [[""] + numeric_df.columns.tolist()] + [[col] + [pearsonr(numeric_df[col], numeric_df[c2])[0] if col != c2 else 1.0 for c2 in numeric_df.columns] for col in numeric_df.columns]
                num_corr_cols = len(numeric_df.columns) + 1
                col_width = min(80, PAGE_WIDTH / num_corr_cols) if num_corr_cols > 0 else PAGE_WIDTH
                story.append(Table(corr_data, colWidths=[col_width]*num_corr_cols, style=[('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('FONTSIZE', (0,0), (-1,-1), 7), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 12))

    # Step 3: Missing Values
    story.append(Paragraph("3. Dealing with Missing Values", styles['SectionTitle']))
    story.append(Paragraph(STEP_DESCRIPTIONS[3], styles['Description']))
    missing_data = session_df.isnull().sum().to_frame('Missing Values').reset_index()
    story.append(Table([missing_data.columns.tolist()] + missing_data.values.tolist(), colWidths=[270, 270], style=[('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('FONTSIZE', (0,0), (-1,-1), 7), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 12))

    # Step 4: Outliers
    story.append(Paragraph("4. Outliers Handling", styles['SectionTitle']))
    story.append(Paragraph(STEP_DESCRIPTIONS[4], styles['Description']))
    outliers_info = {}
    for col in session_df.select_dtypes(include=np.number).columns:
        z_scores = np.abs((session_df[col] - session_df[col].mean()) / session_df[col].std())
        outliers_info[col] = session_df[col][z_scores > 3].count()
    outliers_data = [["Column", "Outliers Count"]] + [[k, v] for k, v in outliers_info.items()]
    story.append(Table(outliers_data, colWidths=[270, 270], style=[('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('FONTSIZE', (0,0), (-1,-1), 7), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(Spacer(1, 12))

    # Step 5: Normalization
    story.append(Paragraph("5. Normalization", styles['SectionTitle']))
    story.append(Paragraph(STEP_DESCRIPTIONS[5], styles['Description']))
    numeric_cols = session_df.select_dtypes(include=np.number).columns
    if numeric_cols.any():
        col = numeric_cols[0]
        norm_df = session_df[[col]].copy()
        norm_df[f'{col}_normalized'] = (session_df[col] - session_df[col].min()) / (session_df[col].max() - session_df[col].min())
        norm_sample = norm_df.head()
        story.append(Table([norm_sample.columns.tolist()] + norm_sample.values.tolist(), colWidths=[270, 270], style=[('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('FONTSIZE', (0,0), (-1,-1), 7), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    else:
        story.append(Paragraph("No numeric columns available.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Step 6: Feature Selection
    story.append(Paragraph("6. Feature Selection", styles['SectionTitle']))
    story.append(Paragraph(STEP_DESCRIPTIONS[6], styles['Description']))
    mi_features, chi2_features = feature_selection(session_df)['mi_features'], feature_selection(session_df)['chi2_features']
    story.append(Paragraph("Mutual Information Top Features: " + ", ".join(mi_features), styles['Normal']))
    story.append(Paragraph("Chi-Squared Top Features: " + ", ".join(chi2_features), styles['Normal']))
    if numeric_cols.any() and len(numeric_cols) > 1:
        X = session_df[numeric_cols].fillna(0)
        y = session_df.iloc[:, 0].fillna(0)
        selector = SelectKBest(mutual_info_classif, k=min(5, len(numeric_cols))).fit(X, y)
        plt.figure(figsize=(8, 4))
        plt.bar(X.columns, selector.scores_)
        plt.title("Feature Importance (Mutual Information)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_FOLDER, 'feature_importance.png'))
        plt.close()
        story.append(Image(os.path.join(STATIC_FOLDER, 'feature_importance.png'), width=250, height=125))
    story.append(Spacer(1, 12))

    # Step 7: Graphic Displays
    story.append(Paragraph("7. Graphic Displays", styles['SectionTitle']))
    story.append(Paragraph(STEP_DESCRIPTIONS[7], styles['Description']))
    if dataset_type == 'Numerical' and numeric_cols.any():
        col = numeric_cols[0]
        plt.figure()
        plt.boxplot(session_df[col].dropna())
        plt.title(f'Boxplot of {col}')
        plt.savefig(os.path.join(STATIC_FOLDER, 'boxplot.png'))
        plt.close()
        story.append(Image(os.path.join(STATIC_FOLDER, 'boxplot.png'), width=250, height=125))
        plt.figure()
        plt.hist(session_df[col].dropna(), bins=50)
        plt.title(f'Histogram of {col}')
        plt.savefig(os.path.join(STATIC_FOLDER, 'histogram.png'))
        plt.close()
        story.append(Image(os.path.join(STATIC_FOLDER, 'histogram.png'), width=250, height=125))
    elif dataset_type == 'NLP/Text':
        text_cols = session_df.select_dtypes(include='object').columns
        if text_cols.any():
            wordcloud = WordCloud(width=800, height=400).generate(' '.join(session_df[text_cols[0]].dropna()))
            wordcloud.to_file(os.path.join(STATIC_FOLDER, 'wordcloud.png'))
            story.append(Image(os.path.join(STATIC_FOLDER, 'wordcloud.png'), width=250, height=125))

    doc.build(story, onFirstPage=add_header, onLaterPages=add_header)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="ExploreX_Report.pdf", mimetype='application/pdf')

def run_all_eda_steps(df, dataset_type):
    return {
        'step1': load_data(df),
        'step2': data_exploration(df),
        'step3': missing_values(df),
        'step4': outliers_handling(df),
        'step5': normalization(df),
        'step6': feature_selection(df),
        'step7': graphic_displays(df, dataset_type),
        'summary': data_summary(df)
    }

def load_data(df):
    return {'size_info': f"The dataset has {len(df.columns)} features and {len(df.index)} tuples."}

def data_exploration(df):
    features = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    first_10 = df.head(10).to_html(index=False, classes='table table-striped table-hover')
    stats = df.describe().to_html(classes='table table-striped table-hover') if dataset_type == 'Numerical' else "Not applicable for NLP/Text data."
    normality = {}
    correlation = {}
    chi_square = {}
    if dataset_type == 'Numerical':
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            stat, p = shapiro(df[col].dropna())
            normality[col] = {'p_value': p, 'normal': p > 0.05}
        if len(numeric_cols) > 1:
            numeric_df = df[numeric_cols].dropna()  # Drop NaNs across all numeric columns
            if len(numeric_df) >= 2:  # Check for at least 2 rows
                correlation = {col: {c2: pearsonr(numeric_df[col], numeric_df[c2])[0] for c2 in numeric_cols if col != c2} for col in numeric_cols}
            else:
                correlation = {}  # Empty correlation if insufficient data
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) > 1:
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                contingency = pd.crosstab(df[col1], df[col2])
                chi2_stat, p, _, _ = chi2_contingency(contingency)
                chi_square[f"{col1} vs {col2}"] = {'chi2': chi2_stat, 'p_value': p}
    return {
        'features': features, 'dtypes': dtypes, 'first_10': first_10, 'stats': stats,
        'normality': normality, 'correlation': correlation, 'chi_square': chi_square
    }

def missing_values(df):
    missing_values = df.isnull().sum().to_frame('Missing Values').to_html(classes='table table-striped table-hover')
    return {'missing_values': missing_values}

def outliers_handling(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    outliers_info = {}
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers_info[col] = df[col][z_scores > 3].count()
    return {'outliers_info': outliers_info}

def normalization(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    normalized_sample = "No numeric columns available."
    if numeric_cols.any():
        col = numeric_cols[0]
        if f'{col}_normalized' in df.columns:
            normalized_sample = df[[col, f'{col}_normalized']].head().to_html(classes='table table-striped table-hover')
    return {'normalized_sample': normalized_sample}

def feature_selection(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    mi_features = chi2_features = ["Not enough data for feature selection."]
    feature_scores = {}
    if len(numeric_cols) > 1 and df.shape[0] > 1:
        X = df[numeric_cols].fillna(0)
        y = df.iloc[:, 0].fillna(0)
        try:
            selector_mi = SelectKBest(mutual_info_classif, k=min(5, len(numeric_cols))).fit(X, y)
            mi_features = X.columns[selector_mi.get_support()].tolist()
            feature_scores = {col: score for col, score in zip(X.columns, selector_mi.scores_)}
            selector_chi2 = SelectKBest(chi2, k=min(5, len(numeric_cols))).fit(X.abs(), y.abs())
            chi2_features = X.columns[selector_chi2.get_support()].tolist()
        except:
            mi_features = chi2_features = ["Error in feature selection."]
    if feature_scores:
        plt.figure(figsize=(8, 4))
        plt.bar(feature_scores.keys(), feature_scores.values())
        plt.title("Feature Importance (Mutual Information)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_FOLDER, 'feature_importance.png'))
        plt.close()
    return {
        'mi_features': mi_features,
        'chi2_features': chi2_features,
        'feature_importance_path': 'feature_importance.png' if feature_scores else None
    }

def graphic_displays(df, dataset_type):
    numeric_cols = df.select_dtypes(include=np.number).columns
    boxplot_path = hist_path = wordcloud_path = None
    
    if dataset_type == 'Numerical' and numeric_cols.any():
        col = numeric_cols[0]
        plt.figure()
        plt.boxplot(df[col].dropna())
        plt.title(f'Boxplot of {col}')
        boxplot_path = os.path.join(STATIC_FOLDER, 'boxplot.png')
        plt.savefig(boxplot_path)
        plt.close()

        plt.figure()
        plt.hist(df[col].dropna(), bins=50)
        plt.title(f'Histogram of {col}')
        hist_path = os.path.join(STATIC_FOLDER, 'histogram.png')
        plt.savefig(hist_path)
        plt.close()
    
    elif dataset_type == 'NLP/Text':
        text_cols = df.select_dtypes(include='object').columns
        if text_cols.any():
            text_col = text_cols[0]
            wordcloud = WordCloud(width=800, height=400).generate(' '.join(df[text_col].dropna()))
            wordcloud_path = os.path.join(STATIC_FOLDER, 'wordcloud.png')
            wordcloud.to_file(wordcloud_path)
    
    return {'boxplot': boxplot_path, 'histogram': hist_path, 'wordcloud': wordcloud_path}

def data_summary(df):
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'missing': df.isnull().sum().sum(),
        'numeric': len(df.select_dtypes(include=np.number).columns),
        'categorical': len(df.select_dtypes(include='object').columns)
    }

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)