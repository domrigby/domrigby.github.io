import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# 1. load your data (expects 'title', 'summary', 'date_read')
with open('data_analysis/paper_one_sentence_descriptions.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['date_read'] = pd.to_datetime(df['date_read'])

# 2. daily counts
daily_counts = (
    df
    .groupby(df['date_read'].dt.date)
    .size()
    .reset_index(name='count')
    .sort_values('date_read')
)

# 3. cumulative
daily_counts['cumulative'] = daily_counts['count'].cumsum()
daily_counts.rename(columns={'date_read': 'Date', 'cumulative': 'Cumulative Papers'}, inplace=True)

# 4. calculate average papers per day
total_papers = daily_counts['Cumulative Papers'].iloc[-1]
date_range_days = (pd.to_datetime(daily_counts['Date']).max() - pd.to_datetime(daily_counts['Date']).min()).days + 1
avg_per_day = total_papers / date_range_days

# 5. plot
fig = px.line(
    daily_counts,
    x='Date',
    y='Cumulative Papers',
    title='<b>Cumulative Number of Papers Read Over Time</b>',
    labels={'Date': '<b>Date</b>', 'Cumulative Papers': '<b>Cumulative Papers</b>'}
)

fig.update_layout(
    template='plotly_white',
    title_x=0.5
)

# 6. add annotation for avg papers per day, total papers, and number of days
fig.add_annotation(
    text=(
        f"<b>Avg:</b> {avg_per_day:.2f} papers/day<br>"
        f"<b>Total:</b> {total_papers} papers<br>"
        f"<b>Days:</b> {date_range_days}"
    ),
    xref="paper", yref="paper",
    x=0.95, y=0.05,
    showarrow=False,
    font=dict(size=16),
    bordercolor="black",
    borderwidth=1,
    borderpad=5,
    bgcolor="white",
    opacity=0.9
)


# 7. save to standalone HTML
fig.write_html("cumulative_papers_read.html", include_plotlyjs='cdn')
fig.show()
