import json

import run

from voicevox_engine.synthesis_engine import SynthesisEngine

if __name__ == "__main__":
    app = run.generate_app(SynthesisEngine(speakers=""))
    with open("docs/api/index.html", "w") as f:
        f.write(
            """<!DOCTYPE html>
<html lang="ja">
<head>
    <title>voicevox_engine API Document</title>
    <meta charset="utf-8">
    <link rel="shortcut icon" href="https://voicevox.hiroshiba.jp/favicon-32x32.png">
</head>
<body>
    <div id="redoc-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js"></script>
    <script>
        Redoc.init(%s, {"hideHostname": true}, document.getElementById("redoc-container"));
    </script>
</body>
</html>"""
            % json.dumps(app.openapi())
        )
