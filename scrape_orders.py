import os
import logging
import requests
import boto3
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [scrape_orders] %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://www.scd.uscourts.gov/mdl-2873/orders.asp"
S3_BUCKET = "generate-input-f5bef08a-9228-4f8c-a550-56d842b94088"
USER_ID = "00000"
PROJECT_ID = "22222"
MAX_WORKERS = 10
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class OrderScraper:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")

    def get_pdf_links(self) -> List[Dict[str, str]]:
        """
        Scrape PDF links from the Case Management Orders page.
        """
        try:
            response = self.session.get(BASE_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if href.lower().endswith('.pdf'):
                    title = link.get_text(strip=True) or Path(href).stem
                    full_url = urljoin(BASE_URL, href)
                    pdf_links.append({
                        'url': full_url,
                        'title': title
                    })
            
            logger.info(f"Found {len(pdf_links)} PDF links")
            return pdf_links
            
        except Exception as e:
            logger.error(f"Error scraping PDF links: {str(e)}")
            return []

    def download_pdf(self, pdf_info: Dict[str, str]) -> Optional[Tuple[str, bytes]]:
        """
        Download a single PDF file.
        """
        try:
            url = pdf_info['url']
            title = pdf_info['title']
            
            logger.info(f"Downloading PDF: {title}")
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            # Generate a filename based on the title
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d')}.pdf"
            
            # Save to temporary file
            temp_path = os.path.join(self.temp_dir, filename)
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Read the file for upload
            with open(temp_path, 'rb') as f:
                content = f.read()
            
            logger.info(f"Successfully downloaded: {filename}")
            return filename, content
            
        except Exception as e:
            logger.error(f"Error downloading PDF {pdf_info['title']}: {str(e)}")
            return None

    def upload_to_s3(self, filename: str, content: bytes) -> bool:
        """
        Upload a PDF file to S3.
        """
        try:
            # Generate S3 key
            s3_key = f"{USER_ID}/{PROJECT_ID}/input/{filename}"
            
            # Calculate MD5 hash for integrity check
            content_md5 = hashlib.md5(content).hexdigest()
            
            # Upload to S3 with metadata
            self.s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=content,
                ContentType='application/pdf',
                Metadata={
                    'md5': content_md5,
                    'source': BASE_URL,
                    'upload_date': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Successfully uploaded to S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return False

    def process_pdf(self, pdf_info: Dict[str, str]) -> bool:
        """
        Process a single PDF: download and upload to S3.
        """
        try:
            result = self.download_pdf(pdf_info)
            if result:
                filename, content = result
                return self.upload_to_s3(filename, content)
            return False
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_info['title']}: {str(e)}")
            return False

    def cleanup(self):
        """
        Clean up temporary files.
        """
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
                os.rmdir(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def run(self):
        """
        Main execution method.
        """
        try:
            # Get PDF links
            pdf_links = self.get_pdf_links()
            if not pdf_links:
                logger.error("No PDF links found")
                return
            
            # Process PDFs in parallel
            successful = 0
            failed = 0
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_pdf = {executor.submit(self.process_pdf, pdf): pdf for pdf in pdf_links}
                
                for future in as_completed(future_to_pdf):
                    pdf = future_to_pdf[future]
                    try:
                        if future.result():
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Error processing {pdf['title']}: {str(e)}")
                        failed += 1
            
            logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")
            
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
        finally:
            self.cleanup()

def main():
    """
    Main entry point.
    """
    try:
        scraper = OrderScraper()
        scraper.run()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
