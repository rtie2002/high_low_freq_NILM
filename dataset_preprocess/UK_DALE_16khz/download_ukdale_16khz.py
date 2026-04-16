import os
import requests
from bs4 import BeautifulSoup

def download_files(url, target_dir):
    """
    Mirrors a directory from CEDA archive containing .flac files.
    Skips already downloaded files.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    print(f"Fetching file list from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')
    
    file_links = []
    for link in links:
        href = link.get('href')
        if href and href.endswith('.flac'):
            full_url = requests.compat.urljoin(url, href)
            file_name = href.split('/')[-1]
            file_links.append((full_url, file_name))
    
    total = len(file_links)
    print(f"Found {total} .flac files to download.")
    
    for i, (f_url, f_name) in enumerate(file_links, 1):
        target_path = os.path.join(target_dir, f_name)
        
        # Check if file exists and has size (rough check for completion)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"[{i}/{total}] Skipping {f_name} (already exists)")
            continue
            
        print(f"[{i}/{total}] Downloading {f_name}...")
        try:
            with requests.get(f_url, stream=True) as r:
                r.raise_for_status()
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"Error downloading {f_name}: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)

if __name__ == "__main__":
    # Target URL for House 2, 2013, Week 30
    DATA_URL = "https://dap.ceda.ac.uk/edc/d1/887733b3-4c04-471f-9404-9f7459c4a1a0/data/version_0/house_2/2013/wk30/"
    
    # Use current directory as target
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    download_files(DATA_URL, CURRENT_DIR)
