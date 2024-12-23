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
        dict: Dictionary containing structured medical information
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
        result = {
            'symptoms': [],
            'lab_results': [],
            'diagnostic_procedures': [],
            'treatments': [],
            'risk_factors': [],
            'complications': [],
            'prevention': [],
            'when_to_see_doctor': []
        }
        
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
        
        # Process each section
        section_elements = soup.find_all('section')
        for section in section_elements:
            title_elem = section.find(['h2', 'h3'])
            if not title_elem:
                continue
                
            section_title = title_elem.get_text(strip=True).lower()
            section_content = []
            
            # Find the section body
            section_body = section.find('div', {'class': 'section-body'})
            if not section_body:
                continue
            
            # Extract bullet points
            bullet_list = section_body.find('ul', {'class': 'bulletlist'})
            if bullet_list:
                for li in bullet_list.find_all('li', recursive=False):
                    content = li.get_text(strip=True)
                    if content:
                        section_content.append(content)
            
            # Map section content to appropriate category
            if 'symptom' in section_title or 'sign' in section_title:
                result['symptoms'].extend(section_content)
            elif 'test' in section_title or 'lab' in section_title:
                result['lab_results'].extend(section_content)
            elif 'diagnos' in section_title:
                result['diagnostic_procedures'].extend(section_content)
            elif 'treatment' in section_title or 'therap' in section_title:
                result['treatments'].extend(section_content)
            elif 'risk' in section_title or 'cause' in section_title:
                result['risk_factors'].extend(section_content)
            elif 'complication' in section_title:
                result['complications'].extend(section_content)
            elif 'prevent' in section_title:
                result['prevention'].extend(section_content)
            elif 'when to' in section_title or 'call' in section_title:
                result['when_to_see_doctor'].extend(section_content)
        
        # Remove empty lists from result
        result = {k: v for k, v in result.items() if v}
        
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
