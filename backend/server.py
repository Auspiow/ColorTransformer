from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder="assets/plots")
CORS(app)  # 允许前端访问

# 读取图像列表
@app.route("/api/plots")
def get_plot_list():
    import os
    files = os.listdir("assets/plots")
    files = [f for f in files if f.endswith(".png")]
    return jsonify(files)

# 访问单个 PNG
@app.route("/plots/<filename>")
def get_plot(filename):
    return send_from_directory("assets/plots", filename)

if __name__ == "__main__":
    app.run(port=8000, debug=True)
