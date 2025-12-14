import ssl, certifi, urllib.request, nltk, os

# Use certifi CA certs for SSL
ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
)

# Make sure local target folder exists
os.makedirs("nltk_data", exist_ok=True)

# Download directly into ./nltk_data inside your project
for pkg in ["stopwords", "punkt", "punkt_tab"]:
    print(f"Downloading {pkg}...")
    nltk.download(pkg, download_dir="nltk_data")
