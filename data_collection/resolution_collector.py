from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import logging
import os
import json
from urllib.parse import urlencode

@dataclass
class CentralBankResolution:
    title: str
    content: str
    url: str
    publication_date: str
    collection_date: str

def setup_chrome_driver() -> webdriver.Chrome:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    service = Service("/usr/local/bin/chromedriver")
    return webdriver.Chrome(service=service, options=chrome_options)

def extract_resolution_data(driver: webdriver.Chrome, resolution_url: str) -> Optional[CentralBankResolution]:
    try:
        driver.get(resolution_url)
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "corpoNormativo")))

        title_element = driver.find_element(By.CLASS_NAME, "titulo-pagina")
        content_elements = driver.find_elements(By.XPATH, "//div[@class='corpoNormativo']//span")

        if not title_element or not content_elements:
            logging.error(f"Missing elements on page: {resolution_url}")
            return None

        title = title_element.text
        content = "\n".join(p.text for p in content_elements)
        publication_date = title[-10:]

        return CentralBankResolution(
            title=title,
            content=content,
            url=resolution_url,
            publication_date=publication_date,
            collection_date=datetime.now().isoformat()
        )
    except Exception as e:
        logging.error(f"Error extracting data from {resolution_url}: {e}")
        return None

def collect_central_bank_resolutions(save_dir: str) -> None:
    logging.info("Starting resolution collection...")
    with setup_chrome_driver() as driver:
        try:
            base_url = "https://www.bcb.gov.br/estabilidadefinanceira/buscanormas"
            params = {
                "dataInicioBusca": "01/01/2020",
                "dataFimBusca": "31/12/2024",
                "tipoDocumento": "Resolução BCB"
            }
            start_row = 0
            all_links = []

            while True:
                params["startRow"] = start_row
                url = f"{base_url}?{urlencode(params)}"
                driver.get(url)
                wait = WebDriverWait(driver, 20)
                wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "resultado-item")))

                resolution_links = [element.get_attribute("href") for element in driver.find_elements(By.XPATH, "//a[contains(@href, 'exibenormativo')]")]
                if not resolution_links:
                    break

                all_links.extend(resolution_links)
                logging.info(f"Collected {len(resolution_links)} links from page starting at row {start_row}")

                start_row += 15

            all_data = []
            for resolution_url in all_links:
                logging.info(f"Extracting data from {resolution_url}...")
                data = extract_resolution_data(driver, resolution_url)
                if data:
                    all_data.append(data)

            save_data(all_data, save_dir)

        except Exception as e:
            logging.error(f"An error occurred during resolution collection: {e}")

def save_data(data, save_dir):
    """Save extracted data to a file."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "resolutions_data.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([d.__dict__ for d in data], f, ensure_ascii=False, indent=4)

    logging.info(f"Data saved to {file_path}") 