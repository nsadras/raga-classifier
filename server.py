import http.server
import socketserver
import glob
import os
from pathlib import Path
from RangeHTTPServer import RangeRequestHandler

PORT = 8000

class DynamicDashboardHandler(RangeRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        # Serve dynamic index for root or index.html
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Dynamic Scan
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
                    .refresh { float: right; font-size: 0.8em; color: #aaa; cursor: pointer; }
                </style>
            </head>
            <body>
                <h1>
                    Available Interactive Plots
                    <span class="refresh" onclick="location.reload()">&#x21bb; Refresh</span>
                </h1>
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
            self.wfile.write(html.encode("utf-8"))
        else:
            # Default static file serving
            super().do_GET()

if __name__ == "__main__":
    # Allow address reuse prevents "Address already in use" errors on restart
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DynamicDashboardHandler) as httpd:
        print(f"Serving Raga Dashboard at http://localhost:{PORT}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
