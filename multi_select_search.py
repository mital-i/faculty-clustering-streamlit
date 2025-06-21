# mesh_selector.py

import streamlit as st
import numpy as np
import pandas as pd
import re
from streamlit_plotly_events import plotly_events
import plotly.express as px

def to_list(cell):
    if isinstance(cell, list):
        return cell
    return [t.strip() for t in re.split(r"[;,]", str(cell)) if t.strip()]

def render_mesh_selector(umap_df, mesh_df):
    """
    Renders:
     1) Two text inputs (faculty name & MeSH term)
     2) One Plotly scatter that respects those filters
     3) Click/box-lasso multi-select
     4) Download of manual or filtered selection
    """

    st.markdown("## 🔍 Filter & Multi-Select Faculty")

    # ————————————————————————————————————————————————————————————————
    # 1) Text inputs
    name_search = st.text_input("Search faculty by name:", "")
    mesh_search = st.text_input("Search by MeSH term:", "")
    cluster_input = st.text_input("Search by cluster number:", "")

    # ————————————————————————————————————————————————————————————————
    # 2) Apply filters to umap_df copy
    df = umap_df.copy()

    # filter by name if requested
    if name_search:
        mask_name = df["Faculty_Full_Name"].str.contains(name_search, case=False)
        df = df[mask_name]
    
    if cluster_input:
        cluster_input_l = cluster_input.strip().lower()
        if "cluster" in df.columns:
            df = df[df["cluster"].astype(str).str.lower().str.contains(cluster_input_l)]


    # build a lookup list of lists from mesh_df
    mesh_lists = (
        mesh_df.set_index("Faculty_Full_Name")["Unique_Mesh_Terms"]
               .apply(to_list)
    )
    # filter by mesh term if requested
    if mesh_search:
        mesh_search_l = mesh_search.strip().lower()
        # find faculty who have that term
        mask_mesh = df["Faculty_Full_Name"].map(lambda name: 
            any(mesh_search_l in term.lower() for term in mesh_lists.get(name, []))
        )
        df = df[mask_mesh]

    df = df.reset_index(drop=True)

    # ————————————————————————————————————————————————————————————————
    # 3) Inject the pretty hover/download string
    df["mesh_str"] = df["Faculty_Full_Name"].map(
        mesh_lists.apply(lambda L: "; ".join(L))
    )
    # ————————————————————————————————————————————————————————————————
    # 4) One Plotly figure + multi-select
    color_sequence = px.colors.qualitative.Set1

    fig = px.scatter(
        df,
        x="V1", y="V2",
        color=umap_df["cluster"].astype(str),
        color_discrete_sequence=color_sequence,
        hover_name="Faculty_Full_Name",
        custom_data=["Faculty_Full_Name", "mesh_str"],
        title="Click or box‐lasso to select points",
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>MeSH terms: %{customdata[1]}<extra></extra>",
    )

    # this both renders the chart & returns click/selection events
    if "clear_lasso" not in st.session_state:
        st.session_state.clear_lasso = False

    # Force re-render with new key if clearing
    if st.session_state.clear_lasso:
        chart_key = f"plotly_chart_{np.random.randint(0, 1e9)}"
    else:
        chart_key = "plotly_chart"

    # Render chart and capture events
    events = plotly_events(
        fig,
        click_event=True,
        select_event=True,
        override_height=600,
        override_width="100%",
        key=chart_key
    )

    # Clear lasso flag to avoid loop
    if st.session_state.clear_lasso:
        st.session_state.clear_lasso = False
        st.experimental_rerun()
    # ————————————————————————————————————————————————————————————————
    # 5) Manage manual selections in session_state
    if "selected_faculty" not in st.session_state:
        st.session_state.selected_faculty = []

    if st.session_state.clear_lasso:
        st.session_state.clear_lasso = False
        st.experimental_rerun()

    for ev in events:
        idx = ev.get("pointIndex")
        if idx is None:
            continue
        name = df.iloc[idx]["Faculty_Full_Name"]
        if name in st.session_state.selected_faculty:
            st.session_state.selected_faculty.remove(name)
        else:
            st.session_state.selected_faculty.append(name)

    # ————————————————————————————————————————————————————————————————
    # 6) Determine which faculty set we’ll download
    sel_manual = st.session_state.selected_faculty
    sel_name   = df["Faculty_Full_Name"].tolist() if (name_search and not sel_manual) else []
    sel_mesh   = df["Faculty_Full_Name"].tolist() if (mesh_search and not sel_manual and not name_search) else []
    sel_cluster = df["Faculty_Full_Name"].tolist() if (cluster_input and not mesh_search and not sel_manual and not name_search) else []
    if sel_manual:
        sel        = sel_manual
        label      = "Download CSV of manually-selected MeSH terms"
        file_name  = "selected_faculty_mesh_terms.csv"
    elif sel_name:
        sel        = sel_name
        label      = f"Download CSV of all “{name_search}” results"
        file_name  = f"name_search_{name_search.replace(' ','_')}_mesh_terms.csv"
    elif sel_mesh:
        sel        = sel_mesh
        label      = f"Download CSV of all “{mesh_search}” results"
        file_name  = f"mesh_search_{mesh_search.replace(' ','_')}_mesh_terms.csv"
    elif sel_cluster:
        sel        = sel_cluster
        label      = f"Download CSV of all cluster “{cluster_input}” results"
        file_name  = f"cluster_search_{cluster_input.replace(' ','_')}_mesh_terms.csv"
    else:
        sel = []

    # ————————————————————————————————————————————————————————————————
    # 7) Render download UI
    if sel:
        st.markdown("### 📥 Download MeSH-terms CSV")
        out = mesh_df[mesh_df["Faculty_Full_Name"].isin(sel)].copy()
        out["Unique_Mesh_Terms"] = out["Unique_Mesh_Terms"]\
                                    .apply(to_list)\
                                    .apply(lambda L: "; ".join(L))
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=label,
            data=csv_bytes,
            file_name=file_name,
            mime="text/csv",
        )
        st.write("**Faculty included:** \n", ", ".join(sel))
        if st.button("Clear selections"):
            st.session_state.selected_faculty.clear()
            st.session_state.clear_lasso = True
            st.experimental_rerun()

    else:
        st.info("No faculty selected—click points or use the search boxes above.")
