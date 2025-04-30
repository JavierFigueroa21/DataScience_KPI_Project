import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

# ==========================================
# Carga de Datos y Resumen General
# ==========================================

def load_dataset(filepath):
    """Carga un dataset desde un archivo CSV."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def basic_summary(df):
    """Imprime un resumen básico del dataset: tamaño, tipos, nulos, duplicados."""
    print("\n=== Dataset Summary ===")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print(f"Data Types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicated Columns: {df.columns[df.columns.duplicated()].tolist()}")

def full_descriptive_statistics(df, rounding=2):
    """Muestra estadísticas descriptivas para variables numéricas y categóricas."""
    print("\n=== Descriptive Statistics for Numerical Variables ===")
    numeric_cols = df.select_dtypes(include=np.number).columns
    print(df[numeric_cols].describe().round(rounding))

    print("\n=== Descriptive Statistics for Categorical Variables ===")
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(f"- Unique values: {df[col].nunique()}")
        print(f"- Most frequent value: {df[col].mode()[0]} ({df[col].value_counts().iloc[0]} times)")
        print(f"- Missing values: {df[col].isnull().sum()}")

def quick_report(df):
    """Genera un reporte rápido con tipos de datos, nulos, valores únicos, mínimos y máximos."""
    dtyp = pd.DataFrame(df.dtypes, columns=['Type'])
    missing = pd.DataFrame(df.isnull().sum(), columns=['Missing'])
    unival = pd.DataFrame(df.nunique(), columns=['Unique'])
    maximo = pd.DataFrame(df.max(numeric_only=True), columns=['Max'])
    minimo = pd.DataFrame(df.min(numeric_only=True), columns=['Min'])
    return dtyp.join(missing).join(unival).join(minimo).join(maximo)

# ==========================================
# Tratamiento de Valores Faltantes
# ==========================================

def missing_values_summary(df):
    """Muestra cuántos valores nulos hay por columna y su porcentaje."""
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_table = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_pct})
    print(missing_table[missing_table['Missing Values'] > 0])

def impute_missing_column(df, col, verbose=True):
    """Imputa valores faltantes de una columna según su tipo y distribución.

    - Categórica: se rellena con la moda.
    - Numérica con baja asimetría (|skew| < 1): media.
    - Numérica con alta asimetría: mediana.
    """
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
        if verbose:
            print(f"Imputed '{col}' with mode.")
    elif np.issubdtype(df[col].dtype, np.number):
        skewness = df[col].skew()
        if abs(skewness) < 1:
            df[col].fillna(df[col].mean(), inplace=True)
            if verbose:
                print(f"Imputed '{col}' with mean (skewness={skewness:.2f}).")
        else:
            df[col].fillna(df[col].median(), inplace=True)
            if verbose:
                print(f"Imputed '{col}' with median (skewness={skewness:.2f}).")
    return df

def intelligent_missing_values_treatment(df, threshold=0.2, verbose=True):
    """Tratamiento automático de valores faltantes a nivel de dataset.

    - Elimina columnas con más del threshold de nulos.
    - Imputa el resto usando `impute_missing_column()`.
    """
    total_rows = len(df)
    missing_count = df.isnull().sum()
    proportions = missing_count / total_rows

    for col in df.columns:
        if missing_count[col] == 0:
            continue
        prop = proportions[col]
        if verbose:
            print(f"\nHandling column: {col} — Missing: {missing_count[col]} ({prop:.2%})")
        if prop > threshold:
            df.drop(columns=[col], inplace=True)
            if verbose:
                print(f"Dropped (>{threshold*100:.0f}% missing)")
        else:
            df = impute_missing_column(df, col, verbose)
    return df

# ==========================================
# Detección y Manejo de Duplicados
# ==========================================

def detect_exact_duplicates(df, verbose=True):
    """Detecta filas duplicadas exactas."""
    duplicates = df[df.duplicated(keep=False)]
    if verbose:
        print(f"\nExact duplicates: {len(duplicates)}")
        print(duplicates.head())
    return duplicates

def remove_exact_duplicates(df):
    """Elimina filas duplicadas exactas."""
    return df.drop_duplicates()

def detect_partial_duplicates(df, subset, verbose=True):
    """Detecta duplicados parciales con base en un subconjunto de columnas."""
    partial_dups = df[df.duplicated(subset=subset, keep=False)]
    if verbose:
        print(f"\nPartial duplicates by {subset}: {partial_dups.shape[0]}")
        print(partial_dups.sort_values(by=subset).head(10))
    return partial_dups

# ==========================================
# Detección y Tratamiento de Outliers
# ==========================================

def visualize_outliers(df, column):
    """Visualiza outliers con histogramas y boxplots."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f"Histogram: {column}")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot: {column}")
    plt.tight_layout()
    plt.show()

def detect_outliers_iqr(df, column, k=1.5, verbose=True):
    """Detecta outliers usando el método IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    mask = (df[column] < lower) | (df[column] > upper)
    if verbose:
        print(f"\nOutliers by IQR: {mask.sum()} | Range: [{lower:.2f}, {upper:.2f}]")
    return mask

def detect_outliers_zscore(df, column, threshold=3, verbose=True):
    """Detecta outliers usando Z-score."""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    mask = z_scores > threshold
    if verbose:
        print(f"\nOutliers by Z-Score: {mask.sum()} (Threshold ±{threshold})")
    return df.loc[mask]

def remove_outliers(df, column, method='iqr', **kwargs):
    """Elimina outliers con IQR o Z-score.

    ⚠️ Requiere revisión visual previa para evitar eliminar datos válidos.
    """
    if method == 'iqr':
        mask = detect_outliers_iqr(df, column, verbose=False, **kwargs)
        df_clean = df[~mask]
        print(f"{mask.sum()} outliers removed from '{column}' ({method}).")
    elif method == 'zscore':
        outliers = detect_outliers_zscore(df, column, verbose=False, **kwargs).index
        df_clean = df.drop(index=outliers)
        print(f"{len(outliers)} outliers removed from '{column}' ({method}).")
    else:
        raise ValueError("Unsupported method.")
    return df_clean

# ==========================================
# Codificación y Escalado de Features
# ==========================================

def one_hot_encode(df, drop_first=False):
    """Aplica One-Hot Encoding a todas las columnas categóricas.

    - Utiliza pd.get_dummies() para crear variables binarias.
    - drop_first=True elimina la primera categoría para evitar multicolinealidad.

    Args:
        df (pd.DataFrame): Dataset de entrada.
        drop_first (bool): Si True, elimina la primera categoría por columna (recomendado en regresión lineal, cambiar por True).

    Returns:
        pd.DataFrame: DataFrame con variables dummy.
    """
    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    print(f"One-hot encoding applied. New shape: {df_encoded.shape}")
    return df_encoded


def encode_categorical_features(df):
    """Codifica variables categóricas usando LabelEncoder."""
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def scale_numeric_features(df):
    """Estandariza variables numéricas con StandardScaler."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# ==========================================
# Visualización de EDA
# ==========================================

def correlation_matrix(df, method='pearson'):
    """Genera y muestra matriz de correlación."""
    corr = df.select_dtypes(include=np.number).corr(method=method)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation Matrix ({method})")
    plt.show()
    return corr

def eda_visualizations(df):
    """Visualiza histogramas y matriz de correlación para EDA inicial."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols].hist(figsize=(12, 10), bins=30)
    plt.tight_layout()
    plt.show()
    correlation_matrix(df)

# ==========================================
# Guardar Dataset Procesado
# ==========================================

def save_processed_dataset(df, filename="processed_dataset.csv"):
    """Guarda el dataset limpio en un archivo CSV."""
    df.to_csv(filename, index=False)
    print(f"Dataset saved as '{filename}'")
