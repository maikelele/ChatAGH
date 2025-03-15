import os
import requests
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random


def is_valid_url(url, domain):
    """Check if the URL belongs to the specified domain."""
    parsed_url = urlparse(url)
    parsed_domain = urlparse(domain)
    return parsed_url.netloc == parsed_domain.netloc or parsed_url.netloc.endswith('.' + parsed_domain.netloc)


def is_target_file(url):
    """Check if the URL points to a PDF, TXT, or DOCX file."""
    extensions = ['.pdf', '.txt', '.docx']
    return any(url.lower().endswith(ext) for ext in extensions)


def get_file_extension(url):
    """Extract file extension from URL."""
    return os.path.splitext(url)[1].lower()


def download_file(url, save_dir):
    """Download a file from URL and save it to the specified directory."""
    try:
        file_name = os.path.basename(urlparse(url).path)
        save_path = os.path.join(save_dir, file_name)

        # Check if file already exists
        if os.path.exists(save_path):
            print(f"File already exists: {file_name}")
            return

        print(f"Downloading: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {file_name}")
        else:
            print(f"Failed to download {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def crawl(url, domain, visited, save_dir, delay):
    """Recursively crawl the domain and download target files."""
    if url in visited:
        return

    visited.add(url)

    try:
        print(f"Crawling: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            print(f"Failed to access {url}, status code: {response.status_code}")
            return

        # If the URL is a target file, download it
        if is_target_file(url):
            download_file(url, save_dir)
            return

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links
        for link in soup.find_all('a', href=True):
            next_url = urljoin(url, link['href'])

            # If it's a target file, download it
            if is_target_file(next_url) and is_valid_url(next_url, domain):
                download_file(next_url, save_dir)
                time.sleep(delay)  # Be nice to the server

            # If it's a valid URL in the same domain, crawl it
            elif is_valid_url(next_url, domain) and next_url not in visited:
                # Add a small random delay to avoid overloading the server
                time.sleep(delay * (1 + random.random()))
                crawl(next_url, domain, visited, save_dir, delay)

    except Exception as e:
        print(f"Error crawling {url}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Crawl a domain and download all PDF, TXT, and DOCX files.')
    parser.add_argument('domain', help='Domain to crawl (e.g., https://example.com)')
    parser.add_argument('--output', '-o', default='downloads', help='Directory to save downloaded files')
    parser.add_argument('--delay', '-d', type=float, default=1.0, help='Delay between requests in seconds')
    args = parser.parse_args()

    # Ensure the domain has a scheme
    if not args.domain.startswith(('http://', 'https://')):
        args.domain = 'https://' + args.domain

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Create subdirectories for each file type
    for ext in ['pdf', 'txt', 'docx']:
        subdir = os.path.join(args.output, ext)
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    visited = set()
    print(f"Starting crawler on domain: {args.domain}")
    print(f"Files will be saved to: {os.path.abspath(args.output)}")

    crawl(args.domain, args.domain, visited, args.output, args.delay)
    print(f"Crawling complete. Visited {len(visited)} URLs.")


if __name__ == '__main__':
    main()