import os
from huggingface_hub import snapshot_download

# 1. 强制设置国内镜像源加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 如果你开着 Clash/V2Ray 等梯子软件，请【取消注释】下面这两行，并确认端口号是 7890
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

print("⏳ 正在连接镜像源并下载模型，请稍候...")

try:
    # 开始下载模型
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir="./local_models/all-MiniLM-L6-v2",
        local_dir_use_symlinks=False  # ⚠️ 这句话在 Windows 系统上非常重要，防止软链接报错
    )
    print("✅ 模型下载并保存成功！可以去 local_models 文件夹查看了。")
except Exception as e:
    print(f"❌ 下载失败，请检查网络或代理设置。错误详情：{e}")

# download_ce_model.py
import os
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
    local_dir="./local_models/cross-encoder/ms-marco-MiniLM-L-6-v2",
    local_dir_use_symlinks=False
)

print("CrossEncoder 模型下载完成")