import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import networkx as nx
import leidenalg as la
import igraph as ig
from multi_select_search import render_mesh_selector
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np

# Configuration
config = {
    'file_path': 'mesh_terms_matrix_5yrs_and_keywords.xlsx',
    'pca_components_to_try': range(1, 7),
    'dbscan_eps': 0.05,
    'dbscan_min_samples': 2,
    'anova_alpha': 0.05,
    'top_n_features_to_plot': 10,
    'cluster_output_path': 'Professors_in_clusters.csv',
    'anova_output_path': 'Significant_terms_per_cluster.csv',
    'top_mesh_terms_output_path': 'Top_Mesh_Terms_Per_Professor.csv',
}

# # Helper functions


def load_and_preprocess_data(file_path, index_col='Faculty_Full_Name'):
    raw_data = pd.read_excel(file_path, index_col=index_col)
    raw_data.reset_index(inplace=True)
    faculty_names_df = pd.read_excel(file_path, usecols=['Faculty_Full_Name'])
    feature_matrix = raw_data.drop(columns=['Faculty_Full_Name'])
    raw_data.columns = raw_data.columns.str.replace(
        ' ', '_').str.replace('-', '_').str.replace(',', '_')
    feature_matrix.columns = feature_matrix.columns.str.replace(
        ' ', '_').str.replace('-', '_').str.replace(',', '_')
    return raw_data, feature_matrix, faculty_names_df


def format_mesh_terms(mesh_terms):
    if isinstance(mesh_terms, list) and len(mesh_terms) > 0 and isinstance(mesh_terms[0], list):
        mesh_terms = mesh_terms[0]
    return ', '.join([term.replace('_', ' ') for term in mesh_terms[:5] if isinstance(term, str)])


# Load and preprocess data
raw_data, feature_matrix, faculty_names_df = load_and_preprocess_data(
    config['file_path'])
mesh_term_columns = [col for col in feature_matrix.columns]

# Calculate top MeSH terms
top_mesh_terms_list = []
faculty_names = []
for index, row in raw_data.iterrows():
    professor_name = row['Faculty_Full_Name']
    mesh_term_counts = Counter()
    for term in mesh_term_columns:
        count = row[term]
        if count > 0:
            mesh_term_counts[term] = count
    top_3_terms = [term for term, count in mesh_term_counts.most_common(3)]
    top_mesh_terms_list.append([top_3_terms])
    faculty_names.append(professor_name)

top_mesh_terms_df = pd.DataFrame(
    {'Faculty_Full_Name': faculty_names, 'Top_Mesh_Terms': top_mesh_terms_list})
top_mesh_terms_df.set_index('Faculty_Full_Name', inplace=True)
top_mesh_terms_df.to_csv(config['top_mesh_terms_output_path'])

# Configuration
config = {
    'file_path': 'mesh_terms_matrix_5yrs_and_keywords.xlsx',
    'pca_components_to_try': range(1, 7),
    'dbscan_eps': 0.05,
    'dbscan_min_samples': 2,
    'anova_alpha': 0.05,
    'top_n_features_to_plot': 10,
    'cluster_output_path': 'Professors_in_clusters.csv',
    'anova_output_path': 'Significant_terms_per_cluster.csv',
    'top_mesh_terms_output_path': 'Top_Mesh_Terms_Per_Professor.csv',
    'umap_cache_path': 'cached_umap.joblib'
}

@st.cache_data
def load_and_preprocess_data(file_path, index_col='Faculty_Full_Name'):
    raw_data = pd.read_excel(file_path, index_col=index_col)
    raw_data.reset_index(inplace=True)
    feature_matrix = raw_data.drop(columns=['Faculty_Full_Name'])
    raw_data.columns = raw_data.columns.str.replace(' ', '_').str.replace('-', '_').str.replace(',', '_')
    feature_matrix.columns = feature_matrix.columns.str.replace(' ', '_').str.replace('-', '_').str.replace(',', '_')
    return raw_data, feature_matrix

@st.cache_data
def run_pca(features):
    pca = PCA()
    embeddings = pca.fit_transform(features)
    return pca, embeddings

@st.cache_data
def run_umap(features, n_neighbors=15, min_dist=0.1, metric='cosine'):
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    return umap.fit_transform(features)

@st.cache_data
def calculate_top_mesh_terms(raw_data, mesh_term_columns):
    top_mesh_terms_list = []
    for index, row in raw_data.iterrows():
        mesh_term_counts = Counter({term: row[term] for term in mesh_term_columns if row[term] > 0})
        top_terms = [term.replace('_', ' ') for term, _ in mesh_term_counts.most_common(3)]
        top_mesh_terms_list.append(', '.join(top_terms))
    return pd.DataFrame({
        'Faculty_Full_Name': raw_data['Faculty_Full_Name'],
        'Top_Mesh_Terms': top_mesh_terms_list
    })

@st.cache_data
def perform_anova(feature_matrix, cluster_labels):
    feature_matrix = feature_matrix.copy()
    feature_matrix['cluster'] = cluster_labels
    p_values = {}
    for feature in feature_matrix.columns[:-1]:
        try:
            model = ols(f"{feature} ~ C(cluster)", data=feature_matrix).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_values[feature] = anova_table["PR(>F)"][0]
        except:
            p_values[feature] = 1.0
    feature_list = list(p_values.keys())
    pval_list = list(p_values.values())
    _, adjusted_p, _, _ = multipletests(pval_list, method='fdr_bh')
    return pd.DataFrame({
        'Feature': feature_list,
        'p_value': pval_list,
        'adjusted_p_values': adjusted_p,
        'significant': adjusted_p < config['anova_alpha']
    }).sort_values('adjusted_p_values')

# Load data
raw_data, feature_matrix = load_and_preprocess_data(config['file_path'])
mesh_term_columns = feature_matrix.columns.tolist()

# PCA
pca, pca_embeddings = run_pca(feature_matrix)

# Streamlit UI
st.title("Faculty Clustering and MeSH Term Analysis")

st.subheader("PCA Explained Variance")
st.line_chart(np.cumsum(pca.explained_variance_ratio_))

# UMAP 2D
st.subheader("UMAP 2D Projection")
st.write("""
**Plot Description:** This plot shows a 2D UMAP representation of the original high-dimensional data.
It aims to preserve both local and global structure. Hover over points to see faculty names and top MeSH terms.
""")
umap_embeddings = UMAP().fit_transform(feature_matrix)
umap_embeddings_df = pd.DataFrame(umap_embeddings, columns=["V1", "V2"])
umap_embeddings_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
umap_embeddings_df = umap_embeddings_df.merge(
    top_mesh_terms_df, on='Faculty_Full_Name', how='left')
fig = px.scatter(umap_embeddings_df, x="V1", y="V2", title="UMAP", hover_name="Faculty_Full_Name",
                 hover_data={"V1": False, "V2": False, 'Top_Mesh_Terms': True},
                 width=800, height=800)
fig.update_traces(
    hovertemplate='<b>%{hovertext}</b><br><br>Top Keywords: %{customdata}<extra></extra>',
    customdata=umap_embeddings_df['Top_Mesh_Terms'].apply(format_mesh_terms)
)
st.plotly_chart(fig)

# UMAP with PCA components
st.subheader("UMAP on PCA Components")
num_components = st.slider("Number of PCA components",
                           min_value=1, max_value=10, value=3)
pca_reduced_features = pca_embeddings[:, :num_components]
umap_result = UMAP(random_state=123).fit_transform(pca_reduced_features)
umap_df_pca = pd.DataFrame(umap_result, columns=["V1", "V2"])
umap_df_pca['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']
umap_df_pca = umap_df_pca.merge(
    top_mesh_terms_df, on='Faculty_Full_Name', how='left')
fig = px.scatter(umap_df_pca, x="V1", y="V2", title=f"UMAP on {num_components} PCA Components", hover_name="Faculty_Full_Name",
                 hover_data={"V1": False, "V2": False, 'Top_Mesh_Terms': True},
                 width=800, height=800)
fig.update_traces(
    hovertemplate='<b>%{hovertext}</b><br><br>Top Keywords: %{customdata}<extra></extra>',
    customdata=umap_df_pca['Top_Mesh_Terms'].apply(format_mesh_terms)
)
st.plotly_chart(fig)

n_pca_components = num_components
pca_reduced = pca_embeddings[:, :n_pca_components]

umap_result = run_umap(pca_reduced)
umap_df = pd.DataFrame(umap_result, columns=['V1', 'V2'])
umap_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']

# Top MeSH terms
top_mesh_terms_df = calculate_top_mesh_terms(raw_data, mesh_term_columns)
top_mesh_terms_df.to_csv(config['top_mesh_terms_output_path'], index=False)
umap_df = umap_df.merge(top_mesh_terms_df, on='Faculty_Full_Name', how='left')

# Clustering
st.subheader("Clustering")
clustering_method = st.selectbox("Clustering Method", ["K-means", "DBSCAN", "Leiden"])
if clustering_method == "K-means":
    k = st.slider("# Clusters", 2, 30, 8)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(pca_reduced)
elif clustering_method == "DBSCAN":
    eps = st.slider("DBSCAN eps", 0.01, 1.0, config['dbscan_eps'], 0.01)
    min_samples = st.slider("min_samples", 2, 10, config['dbscan_min_samples'])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pca_reduced)
else:
    resolution = st.slider("Leiden resolution", 0.1, 2.0, 0.8, 0.1)
    knn = NearestNeighbors(n_neighbors=15)
    knn.fit(pca_reduced)
    _, indices = knn.kneighbors(pca_reduced)
    g = nx.Graph()
    for i in range(len(pca_reduced)):
        for j in indices[i]:
            if i != j:
                g.add_edge(i, j)
    g_ig = ig.Graph.from_networkx(g)
    partition = la.find_partition(g_ig, la.CPMVertexPartition, resolution_parameter=resolution)
    labels = partition.membership

umap_df['cluster'] = labels
raw_data['cluster'] = labels

fig = px.scatter(umap_df, x='V1', y='V2', color='cluster', hover_name='Faculty_Full_Name',
                 hover_data={'Top_Mesh_Terms': True}, width=800, height=800)
st.plotly_chart(fig)

# Mesh selector
mesh_df = pd.read_excel("faculty_unique_mesh_terms.xlsx").rename(columns=lambda c: c.strip())
if "Faculty" in mesh_df.columns:
    mesh_df.rename(columns={"Faculty": "Faculty_Full_Name"}, inplace=True)
render_mesh_selector(umap_df, mesh_df)

# ANOVA analysis
st.subheader("Significant Features by Cluster")
anova_results = perform_anova(feature_matrix, labels)
st.dataframe(anova_results[anova_results['significant']][['Feature', 'adjusted_p_values']])
anova_results.to_csv(config['anova_output_path'], index=False)

# Boxplots
st.subheader("Top Feature Distributions")
top_n = st.slider("Top N features to show", 1, 20, config['top_n_features_to_plot'])
top_features = anova_results[anova_results['significant']]['Feature'].head(top_n).tolist()
plot_data = feature_matrix.copy()
plot_data['cluster'] = labels

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()
for i, feature in enumerate(top_features):
    sns.boxplot(x='cluster', y=feature, data=plot_data, ax=axes[i])
    axes[i].set_title(feature[:20])
plt.tight_layout()
st.pyplot(fig)

# Save outputs
umap_df.to_csv(config['cluster_output_path'], index=False)
st.success("Analysis complete and files saved!")
