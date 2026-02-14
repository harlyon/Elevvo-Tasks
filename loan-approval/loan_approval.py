import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Machine Learning Core
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Suppress warnings for clean output
warnings.filterwarnings('ignore')


class LoanApprovalModel:
    """
    A professional-grade implementation for Loan Approval Prediction.
    Includes data cleaning, extensive EDA (Univariate/Bivariate),
    pipeline-based preprocessing, and model evaluation.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        # Aesthetic configuration
        sns.set_theme(style="whitegrid", palette="muted")

    def load_and_clean_data(self):
        """Loads data and performs standard cleaning steps."""
        self.df = pd.read_csv(self.file_path)

        # Professional standard: Strip whitespace from column names and string values
        self.df.columns = self.df.columns.str.strip()
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].str.strip()

        # Feature Selection: loan_id is non-predictive
        if 'loan_id' in self.df.columns:
            self.df.drop(columns=['loan_id'], inplace=True)

        # Target Encoding (Stored for modeling, but we use labels for EDA)
        self.df['loan_status_encoded'] = self.df['loan_status'].map({'Approved': 1, 'Rejected': 0})

        print(f"Data Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
        return self.df

    def perform_univariate_analysis(self):
        """Performs extensive analysis on individual variables."""
        print("\n[Executing Univariate Analysis...]")
        num_cols = self.df.select_dtypes(include=[np.number]).drop(columns=['loan_status_encoded']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        # 1. Numerical Distributions (Histograms + KDE)
        fig, axes = plt.subplots(int(np.ceil(len(num_cols) / 3)), 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.histplot(self.df[col], kde=True, ax=axes[i], color='teal')
            axes[i].set_title(f'Distribution of {col}', fontsize=12)
        # Remove empty subplots
        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

        # 2. Categorical & Target Counts
        fig, axes = plt.subplots(1, len(cat_cols), figsize=(15, 5))
        for i, col in enumerate(cat_cols):
            sns.countplot(data=self.df, x=col, ax=axes[i], order=self.df[col].value_counts().index)
            axes[i].set_title(f'Frequency of {col}', fontsize=12)
            for container in axes[i].containers: axes[i].bar_label(container)
        plt.tight_layout()
        plt.show()

    def perform_bivariate_analysis(self):
        """Performs analysis on relationships between features and target."""
        print("\n[Executing Bivariate Analysis...]")

        # 1. Correlation Heatmap
        plt.figure(figsize=(12, 8))
        corr = self.df.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title("Numerical Feature Correlation Matrix", fontsize=14)
        plt.show()

        # 2. Critical Drivers: CIBIL & Income vs Loan Status
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # CIBIL Score vs Status
        sns.boxplot(data=self.df, x='loan_status', y='cibil_score', ax=axes[0])
        axes[0].set_title('CIBIL Score vs Loan Status', fontsize=13)

        # Income vs Loan Amount colored by Status
        sns.scatterplot(data=self.df, x='income_annum', y='loan_amount', hue='loan_status', alpha=0.6, ax=axes[1])
        axes[1].set_title('Income vs Loan Amount (by Status)', fontsize=13)

        plt.tight_layout()
        plt.show()

    def build_pipeline(self, model_type='decision_tree'):
        """Creates a Scikit-Learn Pipeline for preprocessing and modeling."""
        X = self.df.drop(columns=['loan_status', 'loan_status_encoded'])
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        if model_type == 'logistic_regression':
            model = LogisticRegression(class_weight='balanced', random_state=42)
        else:
            model = DecisionTreeClassifier(class_weight='balanced', random_state=42)

        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    def train_and_tune(self):
        """Splits data and trains the model."""
        X = self.df.drop(columns=['loan_status', 'loan_status_encoded'])
        y = self.df['loan_status_encoded']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.pipeline.fit(self.X_train, self.y_train)
        print("Model Training Complete.")

    def evaluate(self):
        """Performance evaluation with report and confusion matrix."""
        y_pred = self.pipeline.predict(self.X_test)
        print("\n" + "=" * 40)
        print(f" PERFORMANCE REPORT ")
        print("=" * 40)
        print(classification_report(self.y_test, y_pred, target_names=['Rejected', 'Approved']))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred,
                                                display_labels=['Rejected', 'Approved'],
                                                cmap='Greens', ax=ax)
        plt.title('Confusion Matrix')
        plt.grid(False)
        plt.show()


# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    engine = LoanApprovalModel('loan_approval_dataset.csv')
    engine.load_and_clean_data()

    # NEW: Run Extensive Analysis
    engine.perform_univariate_analysis()
    engine.perform_bivariate_analysis()

    # Modeling Phase
    for m_type in ['decision_tree', 'logistic_regression']:
        print(f"\n[Running {m_type.replace('_', ' ').title()} Pipeline...]")
        engine.build_pipeline(model_type=m_type)
        engine.train_and_tune()
        engine.evaluate()