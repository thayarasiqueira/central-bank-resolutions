import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.bcb.gov.br/estabilidadefinanceira/buscanormas?dataInicioBusca=01%2F01%2F2020&dataFimBusca=31%2F12%2F2024&tipoDocumento=Todos"

def download_resolutions(save_dir):
    print("Downloading resolutions from Central Bank website...")

    response = requests.get(BASE_URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=True)

    for link in links:
        href = link["href"]
        if href.endswith(".pdf"):
            pdf_url = href if href.startswith("http") else f"https://www.bcb.gov.br{href}"
            pdf_name = os.path.join(save_dir, pdf_url.split("/")[-1])

            print(f"Downloading {pdf_name}...")
            with requests.get(pdf_url, stream=True) as r:
                r.raise_for_status()
                with open(pdf_name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

    print("Download completed.")
