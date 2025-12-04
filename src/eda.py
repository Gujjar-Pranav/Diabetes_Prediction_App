# src/eda.py
import matplotlib.pyplot as plt
import seaborn as sns
from .data_prep import load_raw_data, clean_data

def run_eda():
    df = load_raw_data()
    df = clean_data(df)

    print("Shape:", df.shape)
    print(df.info())
    print(df.describe().round(3))

    # pairplot
    sns.pairplot(df)
    plt.show()

    # boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df)
    plt.title("Boxplot of numerical features")
    plt.tight_layout()
    plt.show()

    # correlation heatmap
    plt.figure(figsize=(10, 6))
    corr = df.corr().round(3)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_eda()
