import requests
from bs4 import BeautifulSoup
import json
import logging
from urllib.parse import urlparse, urljoin

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[scrape] %(message)s'
)
logger = logging.getLogger(__name__)

def is_valid_medlineplus_url(url):
    """
    Validates if the URL is a MedlinePlus URL.
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if valid MedlinePlus URL, False otherwise
    """
    try:
        parsed = urlparse(url)
        # Update domain validation to handle both old and new URLs
        valid_domains = {'medlineplus.gov', 'www.nlm.nih.gov'}
        
        # Check if it's a valid domain and contains medlineplus in the path
        is_valid = (
            parsed.scheme in ('http', 'https') and
            any(domain in parsed.netloc for domain in valid_domains) and
            ('medlineplus' in parsed.path.lower() or 'medlineplus' in parsed.netloc.lower())
        )
        
        if is_valid:
            logger.info(f"Valid MedlinePlus URL: {url}")
        else:
            logger.warning(f"Invalid MedlinePlus URL: {url}")
            
        return is_valid
        
    except Exception as e:
        logger.error(f"Error validating URL {url}: {str(e)}")
        return False

def normalize_medlineplus_url(url):
    """
    Normalizes MedlinePlus URL to the current format.
    
    Args:
        url (str): URL to normalize
        
    Returns:
        str: Normalized URL
    """
    try:
        # Handle old NIH URLs
        if 'nlm.nih.gov/medlineplus' in url:
            # Extract the path after medlineplus
            path = url.split('medlineplus/')[-1]
            # Remove any extra slashes and ensure .html extension
            path = path.strip('/')
            if not path.endswith('.html'):
                path = f"{path}.html"
            return f'https://medlineplus.gov/{path}'
        return url
    except Exception as e:
        logger.error(f"Error normalizing URL {url}: {str(e)}")
        return url

def scrape_medline_plus(url):
    """
    Scrapes content from MedlinePlus pages and returns structured data.
    
    Args:
        url (str): The URL of the MedlinePlus page to scrape
        
    Returns:
        dict: Dictionary containing page title, summary, and all sections with their content
    """
    try:
        # Validate URL
        if not url or not isinstance(url, str):
            logger.error("Invalid URL: URL must be a non-empty string")
            return None
        
        # Normalize the URL
        url = normalize_medlineplus_url(url)
            
        if not is_valid_medlineplus_url(url):
            logger.error(f"Invalid URL format: {url}")
            return None
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request with timeout
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize the result dictionary
        result = {}
        
        # Get the page title
        title_elem = soup.find('h1', {'class': 'with-also'})
        if title_elem:
            title_text = title_elem.get_text(strip=True)
            if title_text:
                result['title'] = title_text
            
            # Get "Also called" if present
            also_called = soup.find('span', {'class': 'alsocalled'})
            if also_called:
                also_called_text = also_called.get_text(strip=True)
                if also_called_text:
                    result['also_called'] = also_called_text
        
        # Get the summary content
        summary_section = soup.find('div', {'id': 'topic-summary'})
        if summary_section:
            summary_content = []
            # Get all paragraphs and headers
            for elem in summary_section.find_all(['p', 'h3', 'ul']):
                if elem.name == 'h3':
                    summary_content.append(f"\n{elem.get_text(strip=True)}:")
                elif elem.name == 'ul':
                    for li in elem.find_all('li'):
                        summary_content.append(f"• {li.get_text(strip=True)}")
                else:
                    summary_content.append(elem.get_text(strip=True))
            
            summary_text = '\n'.join(summary_content)
            if summary_text.strip():
                result['summary'] = summary_text
        
        # Initialize sections dict
        sections = {}
        
        # Find all sections
        section_elements = soup.find_all('section')
        
        for section in section_elements:
            # Get section title
            title_elem = section.find(['h2', 'h3'])
            if not title_elem:
                continue
                
            section_title = title_elem.get_text(strip=True)
            if 'References and abstracts' in section_title:
                section_title = 'Journal Articles'
            
            # Initialize section content
            section_content = []
            
            # Find the section body
            section_body = section.find('div', {'class': 'section-body'})
            if not section_body:
                continue
                
            # Process bulletlist items if present
            bulletlist = section_body.find('ul', {'class': 'bulletlist'})
            if bulletlist:
                for li in bulletlist.find_all('li', recursive=False):
                    item_parts = []
                    
                    # Get main link and text
                    main_link = li.find('a')
                    if main_link:
                        item_parts.append(main_link.get_text(strip=True))
                    
                    # Get description text
                    desc = li.find('span', {'class': 'desc-text'})
                    if desc:
                        # Get organization info
                        orgs = desc.find('span', {'class': 'orgs'})
                        if orgs:
                            item_parts.append(f"({orgs.get_text(strip=True)})")
                        
                        # Get document type (PDF, etc)
                        doc_type = desc.find('span', {'class': 'desccode'})
                        if doc_type:
                            item_parts.append(f"[{doc_type.get_text(strip=True)}]")
                    
                    # Get "Also in Spanish" text
                    also_lang = li.find('span', {'class': 'also-lang'})
                    if also_lang:
                        item_parts.append(also_lang.get_text(strip=True))
                    
                    if item_parts:
                        section_content.append(' | '.join(item_parts))
            
            # Add section content to result if not empty
            if section_content:
                sections[section_title] = '\n'.join(section_content)
        
        # Only add sections dict if it's not empty
        if sections:
            result['sections'] = sections
        
        return result
    
    except requests.Timeout:
        logger.error("Request timed out")
        return None
    except requests.RequestException as e:
        logger.error(f"Error fetching the webpage: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing the webpage: {str(e)}")
        return None

def main():
    url = "https://medlineplus.gov/diabetes.html"
    result = scrape_medline_plus(url)
    
    if result:
        # Print the result as formatted JSON
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    else:
        logger.error("Failed to scrape the webpage")
        return None

if __name__ == "__main__":
    main()
