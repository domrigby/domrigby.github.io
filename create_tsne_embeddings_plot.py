#!/usr/bin/env python3
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

# ───────────────────────────────────────────────────────────────────────────────
# 1) PARAMETERS
# ───────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path('.')                # repo root
FILE_EXTS = ('.md', '.txt', '.html')
MODEL_NAME = 'all-MiniLM-L6-v2'     # fast & good
TSNE_PERPLEXITY = 15
TSNE_RANDOM_STATE = 42

# ───────────────────────────────────────────────────────────────────────────────
# 2) HELPER: extract first bullet‐list or first paragraph
# ───────────────────────────────────────────────────────────────────────────────
def extract_keypoints(text: str) -> str:
    # collapse whitespace
    text = re.sub(r'\r\n?', '\n', text)
    # try bullet list first
    bullets = re.findall(r'^\s*[-*]\s+(.+)', text, flags=re.MULTILINE)
    if bullets:
        return ' '.join(bullets[:5])
    # else first paragraph
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paras[0] if paras else text[:200]

def load_file(fp: Path) -> str:
    raw = fp.read_text(encoding='utf-8', errors='ignore')
    if fp.suffix == '.html':
        # strip HTML
        soup = BeautifulSoup(raw, 'html.parser')
        return soup.get_text(separator='\n')
    return raw

# ───────────────────────────────────────────────────────────────────────────────
# 3) BUILD dict: filename → keypoints
# ───────────────────────────────────────────────────────────────────────────────
paper_dict = {}
for fp in DATA_DIR.rglob('*'):

    if '.venv' in fp.parts:
        continue

        # 2) skip diary.md explicitly
    if fp.name.lower() == 'diary.md':
        continue

    if fp.name.lower() == 'readme.md':
        continue

    if fp.suffix.lower() in FILE_EXTS:
        txt = load_file(fp)
        kp = extract_keypoints(txt)
        # drop super short
        if len(kp) > 30:
            paper_dict[fp.name] = kp

print(f"Found {len(paper_dict)} files with keypoints.")

# ───────────────────────────────────────────────────────────────────────────────
# 4) EMBEDDINGS → TSNE
# ───────────────────────────────────────────────────────────────────────────────
titles = list(paper_dict.keys())
texts  = [ paper_dict[t] for t in titles ]

print("Loading embedding model…")
model = SentenceTransformer(MODEL_NAME)
print("Computing embeddings…")
embs = model.encode(texts, show_progress_bar=True)

print("Running t-SNE…")
tsne = TSNE(
    n_components=2,
    perplexity=TSNE_PERPLEXITY,
    random_state=TSNE_RANDOM_STATE,
    init='random',
    learning_rate='auto'
)
XY = tsne.fit_transform(embs)

# ───────────────────────────────────────────────────────────────────────────────
# 5) PLOT in OMNI-EPIC style
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14,10))
# color by index (or replace with any scalar)
colors = plt.cm.viridis_r([i/len(XY) for i in range(len(XY))])
sc = ax.scatter(XY[:,0], XY[:,1], c=colors, s=80, edgecolor='k', linewidth=0.5)

# colorbar
sm = plt.cm.ScalarMappable(cmap='viridis_r',
                           norm=plt.Normalize(vmin=0, vmax=len(XY)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Document index", fontsize=12)

# annotate a few “call‐outs”
# here we pick 6 evenly‐spaced indices; replace with your own
for idx in range(len(XY)): #[0, len(XY)//5, 2*len(XY)//5, 3*len(XY)//5, 4*len(XY)//5]:
    x,y = XY[idx]
    title = titles[idx]
    txt   = paper_dict[title][:100] + '…'
    # draw arrow + box
    ax.annotate(
        '', xy=(x,y), xytext=(x+0.5, y+0.5),
        arrowprops=dict(arrowstyle='->', lw=1.2, color='gray')
    )
    ax.text(
        x+0.5, y+0.5, f"{title}\n",
        bbox=dict(boxstyle="round,pad=0.4", fc='white', ec='gray', lw=0.8),
        fontsize=9,
        ha='left', va='bottom'
    )

ax.set_title("t-SNE of Repository Documents", fontsize=16, pad=16)
ax.set_xlabel("t-SNE 1", fontsize=12)
ax.set_ylabel("t-SNE 2", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
