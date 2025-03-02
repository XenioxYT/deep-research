#!/usr/bin/env python3
"""
Analyze a prompt file for duplicated content.
This script helps identify duplicated URLs, findings, and other content in the research data
that is passed to the report generation model.

Usage:
    python analyze_prompt_duplicates.py <prompt_file>
"""

import json
import re
import sys
import os
from collections import defaultdict
from difflib import SequenceMatcher
import time

def clean_string(text):
    """Clean a string for comparison."""
    if not text:
        return ""
    return re.sub(r'\W+', '', str(text)).lower()

def extract_json_from_prompt(prompt_file):
    """Extract the JSON data from a prompt file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the JSON data in the prompt
    json_match = re.search(r'Using the following information:\s*(\{.*\})', content, re.DOTALL)
    if not json_match:
        print("No JSON data found in prompt file")
        return None
    
    try:
        json_str = json_match.group(1)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def analyze_duplicates(data):
    """Analyze the research data for duplicates."""
    if not data:
        return
    
    start_time = time.time()
    print("Starting analysis...")
    
    # Extract research data
    research_data = data.get('research_data', {})
    high_ranking_sources = data.get('high_ranking_sources', {})
    
    # Check for duplicate URLs in iterations
    all_urls = []
    url_counts = defaultdict(int)
    url_to_iterations = defaultdict(list)
    url_to_content_length = defaultdict(int)
    url_to_content = {}
    
    # First pass: collect all URLs and their content
    print("Collecting URLs and content...")
    for i, iteration in enumerate(research_data.get('iterations', [])):
        for finding in iteration.get('findings', []):
            url = finding.get('source', '')
            if url:
                all_urls.append(url)
                url_counts[url] += 1
                url_to_iterations[url].append(i+1)
                
                # Track content length
                content = finding.get('content', '')
                content_len = len(content)
                if content_len > url_to_content_length[url]:
                    url_to_content_length[url] = content_len
                    url_to_content[url] = content
    
    # Check for duplicate URLs in final sources
    for finding in research_data.get('final_sources', []):
        url = finding.get('source', '')
        if url:
            all_urls.append(url)
            url_counts[url] += 1
            
            # Track content length
            content = finding.get('content', '')
            content_len = len(content)
            if content_len > url_to_content_length[url]:
                url_to_content_length[url] = content_len
                url_to_content[url] = content
    
    # Calculate total content size
    total_content_size = sum(url_to_content_length.values())
    
    # Check for duplicate content (optimized)
    print("Analyzing content similarity (this may take a while)...")
    content_similarity = {}
    
    # Only compare URLs with significant content
    urls_with_content = [url for url, content in url_to_content.items() if len(content) > 100]
    
    # Precompute cleaned content for faster comparison
    cleaned_content = {url: clean_string(content) for url, content in url_to_content.items()}
    
    # Compare content similarity
    for i, url1 in enumerate(urls_with_content):
        # Only compare with URLs that haven't been compared yet
        for url2 in urls_with_content[i+1:]:
            # Skip if URLs are the same
            if url1 == url2:
                continue
                
            # Skip if content lengths are very different (optimization)
            len1 = url_to_content_length[url1]
            len2 = url_to_content_length[url2]
            if min(len1, len2) / max(len1, len2) < 0.5:
                continue
            
            # Compare content
            similarity = SequenceMatcher(
                None, 
                cleaned_content[url1],
                cleaned_content[url2]
            ).ratio()
            
            if similarity > 0.7:  # High similarity threshold
                content_similarity[(url1, url2)] = similarity
    
    # Print results
    print("\n=== DUPLICATE ANALYSIS ===\n")
    
    print(f"Total URLs: {len(all_urls)}")
    print(f"Unique URLs: {len(url_counts)}")
    print(f"Duplicate URLs: {len(all_urls) - len(url_counts)}")
    print(f"Total content size: {total_content_size:,} characters")
    
    # Calculate potential savings from deduplication
    duplicate_urls = [url for url, count in url_counts.items() if count > 1]
    duplicate_content_size = sum(url_to_content_length[url] * (url_counts[url] - 1) for url in duplicate_urls)
    print(f"Potential savings from URL deduplication: {duplicate_content_size:,} characters")
    
    # Calculate potential savings from content similarity deduplication
    content_similarity_savings = 0
    for (url1, url2), similarity in content_similarity.items():
        # Estimate savings as the smaller content size * similarity
        smaller_size = min(url_to_content_length[url1], url_to_content_length[url2])
        content_similarity_savings += int(smaller_size * similarity)
    
    print(f"Potential savings from content similarity deduplication: {content_similarity_savings:,} characters")
    print(f"Total potential savings: {duplicate_content_size + content_similarity_savings:,} characters")
    print(f"Percentage of total content: {(duplicate_content_size + content_similarity_savings) / total_content_size * 100:.2f}%")
    
    print("\n=== DUPLICATE URLS ===\n")
    for url, count in sorted(url_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 1:
            print(f"URL: {url}")
            print(f"Count: {count}")
            print(f"Content length: {url_to_content_length[url]:,} characters")
            print(f"Iterations: {url_to_iterations[url]}")
            print()
    
    print("\n=== SIMILAR CONTENT ===\n")
    for (url1, url2), similarity in sorted(content_similarity.items(), key=lambda x: x[1], reverse=True):
        print(f"URL1: {url1}")
        print(f"URL2: {url2}")
        print(f"Similarity: {similarity:.2f}")
        print(f"Content lengths: {url_to_content_length[url1]:,} and {url_to_content_length[url2]:,} characters")
        print()
    
    print("\n=== HIGH RANKING SOURCES ===\n")
    print(f"Total high ranking sources: {len(high_ranking_sources)}")
    for url in high_ranking_sources:
        print(f"URL: {url}")
        print(f"Appears in iterations: {url_to_iterations.get(url, [])}")
        print(f"Total appearances: {url_counts.get(url, 0)}")
        print(f"Content length: {url_to_content_length.get(url, 0):,} characters")
        print()
    
    # Suggest improvements
    print("\n=== RECOMMENDATIONS ===\n")
    if duplicate_urls:
        print("1. Implement URL-based deduplication to remove duplicate URLs")
    if content_similarity:
        print("2. Implement content similarity deduplication to remove similar content")
    if duplicate_content_size + content_similarity_savings > 0.2 * total_content_size:
        print("3. Consider more aggressive deduplication as significant savings are possible")
    
    print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_prompt_duplicates.py <prompt_file>")
        sys.exit(1)
    
    prompt_file = sys.argv[1]
    if not os.path.exists(prompt_file):
        print(f"File not found: {prompt_file}")
        sys.exit(1)
    
    print(f"Analyzing prompt file: {prompt_file}")
    data = extract_json_from_prompt(prompt_file)
    analyze_duplicates(data)

if __name__ == "__main__":
    main() 