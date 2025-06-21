from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, TextInput, Button, DataTable, TableColumn, CDSView, BooleanFilter, CustomJS
from bokeh.models.tools import BoxSelectTool, LassoSelectTool
from bokeh.plotting import figure
import pandas as pd

import pandas as pd
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from collections import Counter

# Load your original data
raw_data = pd.read_excel("mesh_terms_matrix_5yrs_and_keywords.xlsx", index_col='Faculty_Full_Name')
raw_data.reset_index(inplace=True)
feature_matrix = raw_data.drop(columns=['Faculty_Full_Name'])

# Clean column names
feature_matrix.columns = feature_matrix.columns.str.replace(' ', '_').str.replace('-', '_').str.replace(',', '_')
raw_data.columns = raw_data.columns.str.replace(' ', '_').str.replace('-', '_').str.replace(',', '_')

# Run PCA
pca = PCA()
pca_embeddings = pca.fit_transform(feature_matrix)

# Reduce dimensions
num_components = 3
pca_reduced = pca_embeddings[:, :num_components]

# Run UMAP
umap_embeddings = UMAP(random_state=42).fit_transform(pca_reduced)
umap_df = pd.DataFrame(umap_embeddings, columns=['V1', 'V2'])
umap_df['Faculty_Full_Name'] = raw_data['Faculty_Full_Name']

# Compute top MeSH terms
mesh_term_columns = feature_matrix.columns.tolist()
top_mesh_terms_list = []
for index, row_data in raw_data.iterrows():
    mesh_term_counts = Counter({term: row_data[term] for term in mesh_term_columns if row_data[term] > 0})
    top_terms = [term.replace('Normalized', '').replace('_', ' ') for term, _ in mesh_term_counts.most_common(3)]
    top_mesh_terms_list.append(', '.join(top_terms))

top_mesh_terms_df = pd.DataFrame({
    'Faculty_Full_Name': raw_data['Faculty_Full_Name'],
    'Top_Mesh_Terms': top_mesh_terms_list
})

# Merge top MeSH terms into umap_df
umap_df = umap_df.merge(top_mesh_terms_df, on='Faculty_Full_Name', how='left')
umap_df.rename(columns={"Top_Mesh_Terms": "MeSH_Terms"}, inplace=True)

# # Dummy data: replace with your UMAP and mesh data
# umap_df = pd.DataFrame({
#     'Faculty_Full_Name': ['Alice Smith', 'Bob Johnson', 'Carol Lee'],
#     'V1': [1, 2, 3],
#     'V2': [4, 5, 6],
#     'MeSH_Terms': ['Term A; Term B', 'Term C', 'Term D; Term E']
# })

source = ColumnDataSource(umap_df)
selected_source = ColumnDataSource({k: [] for k in umap_df.columns})

# Text input widgets
name_input = TextInput(title="Search Faculty Name")
mesh_input = TextInput(title="Search MeSH Term")

# Scatter plot with select tools
plot = figure(title="UMAP Projection", tools="pan,wheel_zoom,box_select,lasso_select,reset", height=400)
plot.scatter('V1', 'V2', source=source, size=10, selection_color="firebrick")

# Table showing selected data
columns = [
    TableColumn(field="Faculty_Full_Name", title="Faculty Name"),
    TableColumn(field="MeSH_Terms", title="MeSH Terms")
]
table = DataTable(source=selected_source, columns=columns, width=800)

# Update selection based on search
def update_filter():
    name = name_input.value.strip().lower()
    mesh = mesh_input.value.strip().lower()
    mask = umap_df.apply(
        lambda row: (name in row['Faculty_Full_Name'].lower()) and 
                    (mesh in row['MeSH_Terms'].lower()), axis=1)
    source.selected.indices = list(umap_df[mask].index)

name_input.on_change("value", lambda attr, old, new: update_filter())
mesh_input.on_change("value", lambda attr, old, new: update_filter())

# Update selected_source on selection change
def update_selected(attr, old, new):
    selected_data = source.data
    indices = new
    selected = {k: [selected_data[k][i] for i in indices] for k in selected_data}
    selected_source.data = selected

source.selected.on_change("indices", update_selected)

# Download CSV button
download_button = Button(label="Download Selected as CSV", button_type="success")

download_button.js_on_click(CustomJS(args=dict(source=selected_source), code="""
    const data = source.data;
    const keys = Object.keys(data);
    const rows = [];
    for (let i = 0; i < data[keys[0]].length; i++) {
        const row = keys.map(k => `"${data[k][i]}"`).join(",");
        rows.push(row);
    }
    const csv = keys.join(",") + "\\n" + rows.join("\\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.setAttribute("href", url);
    a.setAttribute("download", "selected_faculty.csv");
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
"""))

# Layout
layout = column(
    row(name_input, mesh_input),
    plot,
    table,
    download_button
)

curdoc().add_root(layout)
curdoc().title = "MeSH Term Selector"
