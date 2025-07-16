import os
import json
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

import openai
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

# ─── 0. Configuration ─────────────────────────────────────────────────────────
JSON_PATH   = 'paper_one_sentence_descriptions.json'
PCA_DIMS    = 30
TSNE_PARAMS = dict(n_components=2, random_state=42, perplexity=15)
UMAP_PARAMS = dict(n_components=2, random_state=42, metric='cosine')
N_CLUSTERS  = 6
COLOR_SEQ   = px.colors.qualitative.Plotly

# ─── 1. OpenAI Setup for Manual Summaries ─────────────────────────────────────
with open('open_ai_key.txt', 'r') as f:
    openai.api_key = f.read().strip()

def summarize_all_clusters(cluster_dict):
    """
    cluster_dict: { cluster_id: [title1, title2, ...], ... }
    Returns { cluster_id: theme_label }
    """
    prompt = [
        "Here are paper clusters. For each cluster_id, give a 2-4 word theme label.",
        "Return only valid JSON, mapping IDs (as strings) to labels.",
        "Example output format: (ensure single line response)",
        "{ \"0\": \"Reinforcement Learning\", \"1\": \"Vision Robotics\", ... }",
        ""
    ]
    for cid, titles in cluster_dict.items():
        prompt.append(f'Cluster {cid}:')
        for t in titles:
            prompt.append(f'- {t}')
        prompt.append("")

    # Fallback to manual input if API calls exhausted
    user_resp = input(f"Prompt:\n{chr(10).join(prompt)}\nPlease input JSON response on one line: ")
    return json.loads(user_resp)

# ─── 2. Load Data, Embed, Reduce, Cluster ─────────────────────────────────────
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

titles = [d['title'] for d in data]
descs  = [d['summary'] for d in data]
dates = [d['date_read'] for d in data]

# Encode descriptions
model = SentenceTransformer('all-MiniLM-L6-v2')
emb   = model.encode(descs, convert_to_numpy=True, show_progress_bar=False)

# Dimensionality reduction
emb_pca  = PCA(n_components=PCA_DIMS, random_state=42).fit_transform(emb)
emb_tsne = TSNE(**TSNE_PARAMS).fit_transform(emb_pca)
emb_umap = UMAP(**UMAP_PARAMS).fit_transform(emb_pca)

# Clustering on t-SNE for summary input
labels_tsne = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit_predict(emb_tsne)
tsne_cluster_titles = pd.DataFrame({'cluster': labels_tsne, 'title': titles}) \
    .groupby('cluster')['title'].apply(list).to_dict()

labels_umap = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit_predict(emb_umap)
umap_cluster_titles = pd.DataFrame({'cluster': labels_umap, 'title': titles}) \
    .groupby('cluster')['title'].apply(list).to_dict()

# Get human/LLM-provided theme labels
tsne_cluster_labels = { "0": "LLM Reasoning via RL", "1": "Distributed RL Systems", "2": "Multi-Agent Curriculum Learning", "3": "Open-Ended AI Research", "4": "Foundation & Video Models", "5": "Robotics & Control RL" }
umap_cluster_labels = { "0": "LLM Reasoning with RL", "1": "Distributed RL Systems", "2": "Multi-Agent Reasoning Games", "3": "Open-Ended AI Agents", "4": "Robotics & Curriculum RL", "5": "World Models & Pretraining" }

# tsne_cluster_labels = summarize_all_clusters(tsne_cluster_titles)
# umap_cluster_labels = summarize_all_clusters(umap_cluster_titles)


# ─── 3. Plotting with Convex Hulls and Theme Labels ──────────────────────────
def plot_embedding(emb2, labels, method_name, out_html, cluster_labels):
    df = pd.DataFrame({
        'x': emb2[:, 0],
        'y': emb2[:, 1],
        'cluster': labels.astype(str),
        'title': titles
    })
    df['theme'] = df['cluster'].map(cluster_labels)
    df['date_read'] = dates

    fig = px.scatter(
        df,
        x='x', y='y',
        color='theme',
        hover_data=[],
        custom_data=['title', 'theme'],
        labels={'x': '<b>Dim 1</b>', 'y': '<b>Dim 2</b>'},
        title=f'<b>{method_name} Dimension Reduction of Description Embeddings</b>',
        color_discrete_sequence=COLOR_SEQ,
        template='plotly_white'
    )
    fig.update_layout(title_x=0.5)
    fig.update_traces(
        hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +
                "Cluster: %{customdata[1]}<extra></extra>"
        )
    )

    # Add convex hulls (off by default)
    for i, theme in enumerate(df['theme'].unique()):

        pts = df[df['theme'] == theme][['x', 'y']].values
        if pts.shape[0] < 3:
            continue
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])

        # base color
        base_hex = COLOR_SEQ[i % len(COLOR_SEQ)]
        r, g, b = hex_to_rgb(base_hex)

        fig.add_trace(
            go.Scatter(
                x=hull_pts[:, 0],
                y=hull_pts[:, 1],
                fill='toself',
                fillcolor=f'rgba({r},{g},{b},0.15)',
                line=dict(color=f'rgba({r},{g},{b},1)', width=1),
                hoverinfo='skip',
                # visible='legendonly',    # hide by default
                visible = True,
                legendgroup='hulls',
                showlegend=(i == 0),
                name='Convex Hulls'
            )
        )

    order_df = df.sort_values('date_read')

    fig.add_trace(
        go.Scatter(
            x=order_df['x'],
            y=order_df['y'],
            mode='lines',
            line=dict(dash='dash', width=1),
            hoverinfo='skip',
            visible='legendonly',  # hide by default
            name='Reading Order'
        )
    )

    fig.update_layout(
        legend=dict(
            orientation="h",  # horizontal legend
            yanchor="bottom",  # anchor to the bottom of the legend box
            y=-0.2,  # push it below the x-axis
            xanchor="center",
            x=0.5  # center it
        ),
        margin=dict(
            b=150  # add some extra bottom margin so it doesn’t get cut off
        )
    )

    centroids = (
        df
        .groupby('theme')[['x', 'y']]
        .mean()
        .reset_index()
    )

    for _, row in centroids.iterrows():
        fig.add_annotation(
            x=row['x'], y=row['y'],
            text=row['theme'],
            showarrow=False,
            font=dict(size=16, family="Arial", color="rgba(0,0,0,0.6)"),
            xanchor="center", yanchor="middle",
            opacity=0.8
        )

    # If you’re doing fig.show():
    fig.show(config={'responsive': True})

    # Or if you’re writing HTML:
    fig.write_html(
        out_html,
        include_plotlyjs='cdn',
        config={'responsive': True}
    )
    return fig

# Generate and display plots
fig_tsne = plot_embedding(emb_tsne, labels_tsne, 't-SNE', 'tsne_papers.html', tsne_cluster_labels)
fig_umap = plot_embedding(emb_umap, labels_umap,  'UMAP',  'umap_papers.html', umap_cluster_labels)

fig_tsne.show()
fig_umap.show()
