
import os
from pathlib import Path
import glob

def generate_index():
    plots = sorted(glob.glob("results-*/plots/umap_interactive.json"), reverse=True)
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raga Analysis Dashboard</title>
        <style>
            body { font-family: sans-serif; background: #111; color: #eee; padding: 40px; }
            a { color: #8af; text-decoration: none; font-size: 1.2em; }
            a:hover { text-decoration: underline; }
            .item { background: #222; padding: 20px; margin-bottom: 10px; border-radius: 8px; }
            h1 { border-bottom: 1px solid #444; padding-bottom: 10px; }
            .meta { color: #888; font-size: 0.9em; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h1>Available Interactive Plots</h1>
    """
    
    if not plots:
        html += "<p>No plots found. Run analysis first.</p>"
    
    for p in plots:
        # p is like results-2026-01-21-14-07/plots/umap_interactive.json
        # extract timestamp
        parts = p.split("/")
        res_dir = parts[0] # results-...
        timestamp = res_dir.replace("results-", "")
        
        link = f"viewer.html?file={p}"
        html += f"""
        <div class="item">
            <a href="{link}">Analysis Run: {timestamp}</a>
            <div class="meta">Path: {p}</div>
        </div>
        """
        
    html += "</body></html>"
    
    with open("index.html", "w") as f:
        f.write(html)
    
    print(f"Generated index.html with {len(plots)} plots.")

if __name__ == "__main__":
    generate_index()
