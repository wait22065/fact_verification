
from flask import Flask, render_template, request, jsonify
from src.verifier import FactVerifier
from src.utils import setup_logger
from src import config
import os

app = Flask(__name__)

logger = setup_logger(os.path.join(config.LOG_DIR, "flask_app.log"))
verifier = FactVerifier(logger)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    claim = data.get('claim')
    # 从请求中获取参数，不再覆盖 config 的全局变量
    mode = data.get('mode', 'RAG_COT')
    num_sentences = int(data.get('num_sentences', 3))
    top_k = int(data.get('top_k', 1))
    
    # 将参数显式传给验证器
    prediction, raw_response, evidence = verifier._verify_single_claim(
        claim, 
        num_sentences=num_sentences, 
        mode=mode,
        top_k=top_k
    )
    
    return jsonify({
        'prediction': prediction,
        'reasoning': raw_response,
        'evidence': evidence,
        'status': 'success'
    })

if __name__ == '__main__':
    # 部署时若用 gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 app_flash:app
    app.run(debug=False, host='0.0.0.0', port=5000)