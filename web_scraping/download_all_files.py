import os
import re
import json
import time
import hashlib
import logging
import requests
import argparse
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from pdf2image import convert_from_path
import docx2txt
import fitz  # PyMuPDF
import tempfile
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ContentProcessor:
    """Process different types of content and convert to text"""

    @staticmethod
    def extract_text_from_pdf(file_path):
        """Extract text from PDF using PyMuPDF and OCR when needed"""
        text = ""
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()

            # If page has very little text, it might need OCR
            if len(page_text.strip()) < 50:
                try:
                    # Convert page to image and use OCR
                    pix = page.get_pixmap()
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                        pix.save(temp.name)
                        ocr_text = pytesseract.image_to_string(temp.name)
                        text += ocr_text + "\n\n"
                    os.unlink(temp.name)
                except Exception as e:
                    logger.error(f"OCR failed for page {page_num} in {file_path}: {str(e)}")
                    text += page_text + "\n\n"
            else:
                text += page_text + "\n\n"

        return text

    @staticmethod
    def extract_text_from_docx(file_path):
        """Extract text from DOCX files"""
        try:
            return docx2txt.process(file_path)
        except Exception as e:
            logger.error(f"Failed to extract text from DOCX {file_path}: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_txt(file_path):
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to extract text from TXT {file_path}: {str(e)}")
            return ""


class WebCrawler:
    """Web crawler for extracting content and downloading files"""

    def __init__(self, start_url, output_dir="./output", max_pages=1000, max_depth=10,
                 concurrency=5, delay=0.5, allowed_domains=None):
        # Parse the start URL to get the base domain
        parsed_url = urlparse(start_url)
        self.base_domain = parsed_url.netloc

        self.start_url = start_url
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.concurrency = concurrency
        self.delay = delay
        self.allowed_domains = allowed_domains or [self.base_domain]

        # Create directories
        self.content_dir = os.path.join(output_dir, "content")
        self.files_dir = os.path.join(output_dir, "files")
        self.raw_html_dir = os.path.join(output_dir, "raw_html")

        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.raw_html_dir, exist_ok=True)

        # Track processed URLs and content hashes to detect duplicates
        self.processed_urls = set()
        self.content_hashes = set()
        self.file_urls = set()

        # File extensions to download
        self.file_extensions = ['.pdf', '.docx', '.doc', '.txt', '.rtf']

        # Special URL patterns that might be document downloads
        self.special_url_patterns = [
            r'download\.php',
            r'/download/',
            r'/file/',
            r'/document/',
            r'/get/',
            r'alias='
        ]

        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Results storage
        self.pages = []
        self.downloaded_files = []

    def is_valid_url(self, url):
        """Check if URL should be processed"""
        if not url or not url.startswith('http'):
            return False

        parsed = urlparse(url)

        # Check if domain is allowed
        domain_allowed = any(parsed.netloc == domain or parsed.netloc.endswith('.' + domain)
                             for domain in self.allowed_domains)

        if not domain_allowed:
            return False

        # Skip common non-content URLs
        skip_patterns = [
            r'\.(css|js|jpg|jpeg|png|gif|svg|ico|mp3|mp4|avi|mov)$',
            r'(calendar|login|logout|signin|signout|register|admin)',
            r'(facebook\.com|twitter\.com|linkedin\.com|instagram\.com)'
        ]

        for pattern in skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        return True

    def is_likely_document_url(self, url):
        """Check if URL is likely to be a document download"""
        # Check for file extensions
        if any(url.lower().endswith(ext) for ext in self.file_extensions):
            return True

        # Check for special URL patterns
        if any(re.search(pattern, url, re.IGNORECASE) for pattern in self.special_url_patterns):
            return True

        return False

    def download_file(self, url, depth):
        """Download a file and process its content"""
        if url in self.file_urls:
            return None

        self.file_urls.add(url)

        try:
            logger.info(f"Downloading file: {url}")

            # Make request with stream=True for files
            response = self.session.get(url, stream=True, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to download {url}, status code: {response.status_code}")
                return None

            # Try to determine filename from headers or URL
            content_disposition = response.headers.get('Content-Disposition')
            if content_disposition and 'filename=' in content_disposition:
                filename = re.findall('filename=(.+)', content_disposition)[0].strip('"\'')
            else:
                # Extract filename from URL, handling query parameters
                parsed_url = urlparse(url)
                path = parsed_url.path

                # Handle special case for download.php with alias parameter
                if 'alias=' in url:
                    alias_match = re.search(r'alias=([^&]+)', url)
                    if alias_match:
                        filename = alias_match.group(1)
                    else:
                        filename = os.path.basename(path) or f"document_{hashlib.md5(url.encode()).hexdigest()[:10]}"
                else:
                    filename = os.path.basename(path) or f"document_{hashlib.md5(url.encode()).hexdigest()[:10]}"

            # Clean up the filename
            filename = re.sub(r'[^\w\-\.]', '_', filename)

            # Ensure file has an extension if it's missing
            if not os.path.splitext(filename)[1]:
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' in content_type:
                    filename += '.pdf'
                elif 'word' in content_type or 'docx' in content_type:
                    filename += '.docx'
                elif 'text/plain' in content_type:
                    filename += '.txt'
                else:
                    filename += '.bin'

            file_path = os.path.join(self.files_dir, filename)

            # Save the file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Process file content based on type
            text_content = ""
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext in ['.pdf']:
                text_content = ContentProcessor.extract_text_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                text_content = ContentProcessor.extract_text_from_docx(file_path)
            elif file_ext in ['.txt', '.rtf']:
                text_content = ContentProcessor.extract_text_from_txt(file_path)

            # Save extracted text
            if text_content:
                text_filename = os.path.splitext(filename)[0] + ".md"
                text_path = os.path.join(self.content_dir, text_filename)

                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)

                # Add to results
                self.downloaded_files.append({
                    'url': url,
                    'file_path': file_path,
                    'text_path': text_path,
                    'filename': filename,
                    'file_type': file_ext[1:],  # Remove the dot
                    'depth': depth
                })

                return {
                    'url': url,
                    'filename': filename,
                    'text': text_content
                }

        except Exception as e:
            logger.error(f"Error downloading file {url}: {str(e)}")

        return None

    def extract_content(self, url, html, depth):
        """Extract clean content from HTML, removing duplicates like headers and menus"""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove common non-content elements
        for element in soup.select(
                'header, footer, nav, .menu, .navigation, .sidebar, .footer, .header, .navbar, .nav, aside, .social, .ads, .advertisement, script, style, [role="banner"], [role="navigation"]'):
            element.decompose()

        # Try to find the main content area
        main_content = None
        content_containers = soup.select(
            'main, article, .content, .main, .post, #content, #main, .article, .post-content, [role="main"]')

        if content_containers:
            # Use the largest content container by text length
            main_content = max(content_containers, key=lambda x: len(x.get_text()))
        else:
            # If no clear content container, use the body and remove suspicious elements
            main_content = soup.body

            # Remove elements with very little text but many children (likely menus)
            for element in soup.find_all(['div', 'ul', 'ol']):
                if element.find_all() and len(element.get_text(strip=True)) < 100 and len(element.find_all()) > 5:
                    element.decompose()

        if main_content is None:
            main_content = soup

        # Extract the cleaned text
        text = main_content.get_text(separator='\n').strip()

        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Extract title
        title = soup.title.get_text() if soup.title else urlparse(url).path

        # Hash the content to detect duplicates
        content_hash = hashlib.md5(text.encode()).hexdigest()

        # If this content is a near duplicate, mark it
        is_duplicate = content_hash in self.content_hashes
        if not is_duplicate:
            self.content_hashes.add(content_hash)

        # Extract metadata
        metadata = {
            'url': url,
            'title': title,
            'depth': depth,
            'duplicate': is_duplicate,
            'content_hash': content_hash,
            'word_count': len(text.split())
        }

        # Find all headings and create a structure
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            headings.append({
                'level': int(h.name[1]),
                'text': h.get_text(strip=True)
            })

        metadata['headings'] = headings

        return {
            'metadata': metadata,
            'content': text,
            'html': str(main_content)
        }

    def crawl_url(self, url, depth=0):
        """Crawl a single URL, extract content and find links"""
        if url in self.processed_urls or depth > self.max_depth or len(self.processed_urls) >= self.max_pages:
            return

        self.processed_urls.add(url)
        logger.info(f"Crawling [{depth}/{self.max_depth}] {url}")

        try:
            # Check if this might be a document download URL
            if self.is_likely_document_url(url):
                result = self.download_file(url, depth)
                time.sleep(self.delay)  # Respect rate limiting
                return result

            # Otherwise treat as HTML page
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}, status code: {response.status_code}")
                return

            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                # This might be a file download
                if any(ext[1:] in content_type for ext in self.file_extensions):
                    return self.download_file(url, depth)
                return

            html = response.text

            # Save raw HTML
            page_id = hashlib.md5(url.encode()).hexdigest()[:10]
            raw_path = os.path.join(self.raw_html_dir, f"{page_id}.html")
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(html)

            # Extract clean content
            extracted = self.extract_content(url, html, depth)
            if not extracted or extracted['metadata']['duplicate']:
                return

            # Save as markdown
            clean_filename = f"{page_id}.md"
            clean_path = os.path.join(self.content_dir, clean_filename)

            with open(clean_path, 'w', encoding='utf-8') as f:
                f.write(f"# {extracted['metadata']['title']}\n\n")
                f.write(f"URL: {url}\n")
                f.write(f"Crawled: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(extracted['content'])

                # Save metadata as JSON

                metadata_path = os.path.join(self.content_dir, f"{page_id}_meta.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(extracted['metadata'], f, indent=2)

                # Add to results
                self.pages.append({
                    'url': url,
                    'title': extracted['metadata']['title'],
                    'path': clean_path,
                    'metadata': extracted['metadata']
                })

                # Extract links
                links = []
                soup = BeautifulSoup(html, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href:
                        absolute_url = urljoin(url, href)
                        if self.is_valid_url(absolute_url):
                            links.append(absolute_url)

                # Return data for further processing
                return {
                    'url': url,
                    'links': links,
                    'content': extracted,
                    'depth': depth
                }

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return None
        finally:
            # Respect rate limiting
            time.sleep(self.delay)


    def crawl(self):
        """Main crawling method"""
        start_time = time.time()
        pending_urls = [(self.start_url, 0)]  # (url, depth)
        results = []

        with tqdm(total=self.max_pages) as pbar:
            while pending_urls and len(self.processed_urls) < self.max_pages:
                # Get next batch of URLs
                batch = pending_urls[:self.concurrency]
                pending_urls = pending_urls[self.concurrency:]

                # Process batch in parallel
                with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                    future_to_url = {executor.submit(self.crawl_url, url, depth): url for url, depth in batch}

                    for future in future_to_url:
                        result = future.result()
                        if result:
                            if 'links' in result:  # HTML page
                                # Add new links to pending
                                new_links = [(link, result['depth'] + 1) for link in result['links']]
                                pending_urls.extend(new_links)
                                results.append(result)
                                pbar.update(1)

                # Sort pending URLs by depth (breadth-first approach)
                pending_urls.sort(key=lambda x: x[1])

        end_time = time.time()

        # Generate summary
        summary = {
            'start_url': self.start_url,
            'pages_crawled': len(self.pages),
            'files_downloaded': len(self.downloaded_files),
            'time_taken': end_time - start_time,
            'allowed_domains': self.allowed_domains
        }

        # Save summary
        with open(os.path.join(self.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        # Save pages index
        with open(os.path.join(self.output_dir, 'pages_index.json'), 'w', encoding='utf-8') as f:
            json.dump(self.pages, f, indent=2)

        # Save files index
        with open(os.path.join(self.output_dir, 'files_index.json'), 'w', encoding='utf-8') as f:
            json.dump(self.downloaded_files, f, indent=2)

        logger.info(f"Crawling completed: {len(self.pages)} pages and {len(self.downloaded_files)} files processed")

        return summary


def main():
    """Main function to run the crawler"""
    parser = argparse.ArgumentParser(description='Advanced Web Crawler for RAG Content Extraction')
    parser.add_argument('--url', required=True, help='Starting URL to crawl')
    parser.add_argument('--output', default='./output', help='Output directory')
    parser.add_argument('--max-pages', type=int, default=10000, help='Maximum number of pages to crawl')
    parser.add_argument('--max-depth', type=int, default=100, help='Maximum depth for crawling')
    parser.add_argument('--concurrency', type=int, default=5, help='Number of concurrent requests')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between requests in seconds')
    parser.add_argument('--domains', nargs='+', help='Allowed domains (defaults to domain of start URL)')

    args = parser.parse_args()

    crawler = WebCrawler(
        start_url=args.url,
        output_dir=args.output,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        concurrency=args.concurrency,
        delay=args.delay,
        allowed_domains=args.domains
    )

    summary = crawler.crawl()
    print(f"Crawling completed in {summary['time_taken']:.2f} seconds")
    print(f"Pages crawled: {summary['pages_crawled']}")
    print(f"Files downloaded: {summary['files_downloaded']}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()