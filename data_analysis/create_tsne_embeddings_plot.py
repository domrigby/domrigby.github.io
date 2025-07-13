import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

# 1. your existing data
# 1. Dictionary of papers → one-sentence descriptions
with open('paper_one_sentence_descriptions.json', 'r') as f:
    data = json.load(f)
papers = {entry['title']: entry['summary'] for entry in data}


titles, descs = list(papers), list(papers.values())

# 2. embed & project
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode(descs, convert_to_numpy=True, show_progress_bar=False)
emb2 = TSNE(n_components=2, random_state=42, perplexity=15).fit_transform(emb)

# 3. cluster
labels = KMeans(n_clusters=6, random_state=42).fit_predict(emb2)

# 4. build a DataFrame
df = pd.DataFrame({
    'x': emb2[:,0], 'y': emb2[:,1],
    'cluster': labels.astype(str),
    'title': titles
})

# 5. plotly express!
fig = px.scatter(
    df, x='x', y='y',
    color='cluster',               # color‐by cluster
    hover_name='title',            # show paper title on hover
    labels={'x':'**t-SNE dim 1**','y':'**t-SNE dim 2**'},
    title="t-SNE of Paper Descriptions",
    size_max = 15
)
fig.update_layout(template='plotly_white', title_x=0.5)

# 6. dump as a standalone HTML
fig.write_html("tsne_papers.html", include_plotlyjs='cdn')
