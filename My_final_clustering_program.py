import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
# MinMaxScaler
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    silhouette_score,

)
# from algorithms.nPyGK import nPyGK
from algorithms.nPyGK import nPyGK

# Create output dir
def create_unique_output_dir(output_dir):
    """Create a unique directory to save results."""
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_output_dir = os.path.join(output_dir, f"run_{time_stamp}")
    os.makedirs(unique_output_dir, exist_ok=True)
    return unique_output_dir

# load data and preprocess output data and target
def load_and_preprocess_data(input_csv):
    df = pd.read_csv(input_csv)
    X = df.iloc[:, :-1].values  # Features
    true_labels = df.iloc[:, -1].values  # True labels

    # Encode labels
    label_encoder = LabelEncoder()
    true_labels_encoded = label_encoder.fit_transform(true_labels)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, true_labels_encoded

#  Caltulate all indices
def calculate_indices(X, predicted_labels, cluster_centers,true_label_encoded):
    # Number of samples and clusters
    n_samples = X.shape[0]
    n_clusters = len(cluster_centers)
    # Partition Coefficient (PC)
    pc = np.sum(predicted_labels**2) / n_samples
    
    
    cluster_dists = [
        np.sum((predicted_labels[k,:]**2) * np.linalg.norm(X - cluster_centers[k], axis=1) ** 2)
        for k in range(n_clusters)
    ]
    compactness = np.sum(cluster_dists)

    # Minimum separation distance (between-cluster dispersion)
    separation = np.min([
        np.linalg.norm(cluster_centers[i] - cluster_centers[j])**2
        for i in range(n_clusters)
        for j in range(i + 1, n_clusters)
    ])

    # Xie-Beni Index
    xb = compactness / (n_samples * separation) if separation > 0 else np.inf
    xb = np.minimum(xb,1)

    # Global centroid for Fukuyama-Sugeno Index
    global_centroid = np.mean(X, axis=0)

    # Separation term for Fukuyama-Sugeno Index
    separation_fs = np.sum([
        np.sum(predicted_labels[k,:]**2) * np.linalg.norm(cluster_centers[k] - global_centroid) ** 2
        for k in range(n_clusters)
    ])

    # Fukuyama-Sugeno Index
    se = compactness - separation_fs   
    
    predicted_labels = np.argmax(predicted_labels,axis=0)
    if len(np.unique(predicted_labels)) > 1:
        ss = silhouette_score(X,predicted_labels)
        ars = adjusted_rand_score(predicted_labels, true_label_encoded)
        amis = adjusted_mutual_info_score(predicted_labels, true_label_encoded)
        hh, cc, vv = homogeneity_completeness_v_measure(predicted_labels, true_label_encoded)
    else:
        ss = np.nan
        ars = np.nan
        amis = np.nan
        hh, cc, vv = np.nan,np.nan,np.nan
    return {
        "partition coefficient": pc,
        "fukuyama sugeno index": se,
        "xie beni index": xb,
        "silhouette score": ss,
        "adjusted rand score": ars,
        "adjusted mutual info score": amis,
        "homogeneity": hh,
        "completeness": cc,
        "v measure": vv,
    }


# calculate indices for different n_pyth and alpha and create dataframe
def generate_dataframe(X,true_label_encoded, number_of_clusters, m, MAX_ITER):
    data = []
    # Loop through n_pyth and alpha values and get the results
    for n_pyth in np.arange(1, 6):
        for alpha in np.linspace(0.1, n_pyth, 50):
            nPygk = nPyGK(n_clusters=number_of_clusters, m=m, max_iter=MAX_ITER, n_pyth=n_pyth, alpha=alpha)
            cluster_centers = nPygk.fit(X)
            predicted_labels = nPygk.predict(X)
            metrices = calculate_indices(X, predicted_labels, cluster_centers,true_label_encoded)
            # Add n_pyth, alpha, and dictionary values (a, b, c, ..., g) to the data list
            data.append({'n_pyth': n_pyth, 'alpha': alpha, **metrices})
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)
    return df


# Create latex table
def create_latex_table(df, unique_output_dir):
    """ Generate and save LaTeX table."""
    selected_columns = [
        "n_pyth", 
        "alpha",
        "partition coefficient",
        "fukuyama sugeno index",
        "xie beni index",
        "silhouette score",
        "adjusted rand score",
        "adjusted mutual info score",
        "homogeneity",
        "completeness",
        "v measure"
    ]

    df_selected = df[selected_columns].dropna()
    latex_table = df_selected.to_latex(
        index=False,
        caption="Clustering Metrics for Different Values of n_pyth",
        label="tab:clustering_metrics"
    )

    with open(f"{unique_output_dir}/clustering_metrics_table.tex", "w") as file:
        file.write(latex_table)

    print(f"LaTeX table saved to {unique_output_dir}/clustering_metrics_table.tex")


# Generate graphs
def generate_clustering_graphs(df, unique_output_dir):

    if not os.path.exists(unique_output_dir):
        os.makedirs(unique_output_dir)

    # Metrics to plot and their titles
    metrics = [
        "partition coefficient", "fukuyama sugeno index", "xie beni index",
        "silhouette score", "adjusted rand score", "adjusted mutual info score",
        "homogeneity", "completeness", "v measure"
    ]
    metric_titles = [
        "Partition Coefficient", "Fukuyama Sugeno Index", "Xie-Beni Index",
        "Silhouette Score", "Adjusted Rand Score", "Adjusted Mutual Info Score",
        "Homogeneity", "Completeness", "V-Measure"
    ]

    reference_row = df[(df["n_pyth"] == 1) & (df["alpha"] == 1)].iloc[0]
    # Create a 3x3 grid for the plots
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        for n_pyth, group in df.groupby("n_pyth"):
            axes[i].plot(group["alpha"], group[metric], marker="o", label=f"n={n_pyth}")
        # Add horizontal line for reference value
        ref_value = reference_row[metric]
        axes[i].axhline(y=ref_value, color='red', linestyle='--', label=f"gk_value")
        axes[i].set_title(f"{metric_titles[i]} vs Alpha",fontsize = 20)
        axes[i].set_xlabel("Alpha",fontsize = 18)
        axes[i].set_ylabel(metric_titles[i],fontsize = 18)
        axes[i].legend(fontsize=18)
        axes[i].grid(True)

    # Hide any unused subplots (in case the number of metrics is less than 9)
    for j in range(len(metrics), 9):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{unique_output_dir}/clustering_metrics.pdf",format='pdf')
    plt.close()


def find_max_indices_for_n_pyth(df, unique_output_dir,m):

    # Automatically identify metric columns (excluding n_pyth and alpha)
    df = df.dropna()
    metric_columns = [col for col in df.columns if col not in ["n_pyth", "alpha"]]

    results = []

    # Loop through unique values of n_pyth
    for n_pyth in df["n_pyth"].unique():
        filtered_df = df[df["n_pyth"] == n_pyth]

        # Find the max or min value and corresponding alpha for each metric column
        for col in metric_columns:
            if col in ["xie beni index","fukuyama sugeno index"]:
                # For Xie-Beni index, find the minimum value
                min_row = filtered_df.loc[filtered_df[col].dropna().idxmin()]
                results.append({
                    "n_pyth": n_pyth,
                    "metric": col,
                    "value": min_row[col],
                    "alpha": min_row["alpha"]
                })
            else:
                # For other metrics, find the maximum value
                max_row = filtered_df.loc[filtered_df[col].dropna().idxmax()]
                results.append({
                    "n_pyth": n_pyth,
                    "metric": col,
                    "value": max_row[col],
                    "alpha": max_row["alpha"]
                })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Generate the LaTeX table from the entire DataFrame
    latex_table = results_df.to_latex(
        index=False,  # Do not include row indices
        caption="Best Values of Different Indices for Values of n_pyth and Alpha",  
        label="tab 1: Max indices"  # Optional label for referencing the table
    )

    # Save the LaTeX table to a file
    with open(f"{unique_output_dir}/Clustering_Metrics_Table_maximum.tex", "w") as file:
        file.write(latex_table)

    print(f"LaTeX table saved to {unique_output_dir}/Clustering_Metrics_Table_maximum.tex")

# Main 
def main(input_csv, number_of_clusters, output_dir):
    # Create a unique subdirectory for this run
    unique_output_dir = create_unique_output_dir(output_dir)

    # Load and preprocess data
    X, true_labels_encoded = load_and_preprocess_data(input_csv)

    MAX_ITER = 100
    m = 2.1
    df_pyth = generate_dataframe(X,true_labels_encoded, number_of_clusters, m, MAX_ITER)
    generate_clustering_graphs(df_pyth, unique_output_dir)
    # Create LaTeX table
    create_latex_table(df_pyth, unique_output_dir)
    print(f"All outputs saved in: {unique_output_dir}")
   # Find maximum values of each metrics
    find_max_indices_for_n_pyth(df_pyth, unique_output_dir,m)    