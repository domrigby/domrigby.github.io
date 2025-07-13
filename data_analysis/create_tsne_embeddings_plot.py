import pandas as pd
import numpy as np
import json

import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

from umap.umap_ import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

# ─── 1. Configuration ──────────────────────────────────────────────────────────
JSON_PATH      = 'paper_one_sentence_descriptions.json'
PCA_DIMS       = 30
TSNE_PARAMS    = dict(n_components=2, random_state=42, perplexity=15)
UMAP_PARAMS    = dict(n_components=2, random_state=42, metric='cosine')
N_CLUSTERS     = 6

# pick a consistent colour sequence
COLOR_SEQ = px.colors.qualitative.Plotly

# ─── 2. Load & Embed ───────────────────────────────────────────────────────────
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

titles = [d['title']   for d in data]
descs  = [d['summary'] for d in data]

# 2a. Sentence‑Transformer embeddings
model        = SentenceTransformer('all-MiniLM-L6-v2')
emb_original = model.encode(descs, convert_to_numpy=True, show_progress_bar=False)

# 2b. PCA pre‑reduction
pca     = PCA(n_components=PCA_DIMS, random_state=42)
emb_pca = pca.fit_transform(emb_original)

# 2c. manifold reductions
emb_tsne = TSNE(**TSNE_PARAMS).fit_transform(emb_pca)
emb_umap = UMAP(**UMAP_PARAMS).fit_transform(emb_pca)

# ─── 3. Clustering ─────────────────────────────────────────────────────────────
labels_tsne = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit_predict(emb_tsne)
labels_umap = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit_predict(emb_umap)

# ─── 4. Plotting Helper ───────────────────────────────────────────────────────
def plot_embedding(emb2, labels, method_name, out_html):
    df = pd.DataFrame({
        'x': emb2[:,0],
        'y': emb2[:,1],
        'cluster': labels.astype(str),
        'title': titles
    })

    # base scatter with explicit color sequence
    fig = px.scatter(
        df,
        x='x', y='y',
        color='cluster',
        hover_name='title',
        color_discrete_sequence=COLOR_SEQ,
        labels={'x':'<b>Dim 1</b>', 'y':'<b>Dim 2</b>'},
        title=f'<b>{method_name} of Paper Descriptions</b>',
        template='plotly_white'
    )
    fig.update_layout(title_x=0.5)

    # convex hulls, one legend item "Convex Hulls"
    for i, cl in enumerate(sorted(df['cluster'].unique(), key=int)):
        pts = df[df['cluster']==cl][['x','y']].values
        if pts.shape[0] < 3:
            continue

        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close polygon

        # get the cluster color
        base_hex = COLOR_SEQ[i % len(COLOR_SEQ)]
        r, g, b  = hex_to_rgb(base_hex)

        fig.add_trace(
            go.Scatter(
                x=hull_pts[:,0],
                y=hull_pts[:,1],
                fill='toself',
                fillcolor=f'rgba({r},{g},{b},0.15)',
                line=dict(color=f'rgba({r},{g},{b},1)', width=1),
                hoverinfo='skip',
                visible='legendonly',      # off by default
                legendgroup='hulls',       # group under one legend item
                showlegend=(i==0),         # only first trace shows legend entry
                name='Convex Hulls'
            )
        )

    # export HTML
    fig.write_html(out_html, include_plotlyjs='cdn')
    return fig

# ─── 5. Generate & Show ───────────────────────────────────────────────────────
fig_tsne = plot_embedding(emb_tsne, labels_tsne, 't‑SNE', 'tsne_papers.html')
fig_umap = plot_embedding(emb_umap, labels_umap,  'UMAP',  'umap_papers.html')

# if you're in a notebook or want immediate display:
fig_tsne.show()
fig_umap.show()
