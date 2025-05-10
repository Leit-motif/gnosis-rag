#!/usr/bin/env python
import re
import sys
import json
from pathlib import Path

# Test content with duplicate tags and links
TEST_CONTENT = """---
title: Test Document
tags: [Tag1, tag1, Philosophy, philosophy]
---

# Test Document

This is a document with [[Link1]] and [[link1]] and [[Philosophy]] and [[philosophy]].
It also has #Tag1 #tag1 #Philosophy #philosophy tags.

Let's link to [[Moon]] and [[moon]] and [[Time]] and [[time]].
"""

def extract_and_dedupe_tags(content):
    """Extract and deduplicate tags from content"""
    # Extract tags from frontmatter
    frontmatter_tags = []
    frontmatter_match = re.search(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if frontmatter_match:
        frontmatter = frontmatter_match.group(1)
        print(f"Found frontmatter: {frontmatter}")
        tags_match = re.search(r'tags:\s*\[(.*?)\]', frontmatter)
        if tags_match:
            tags_content = tags_match.group(1)
            print(f"Found tags content: {tags_content}")
            frontmatter_tags = [t.strip() for t in tags_content.split(',')]
            print(f"Extracted frontmatter tags: {frontmatter_tags}")

    # Extract hashtags from content
    content_no_code = re.sub(r'`[^`]*`', '', content)
    content_filtered = re.sub(r'https?://\S+', '', content_no_code)
    hashtag_pattern = r'(?<!\w)#([a-zA-Z0-9_/-]+)'
    content_tags = re.findall(hashtag_pattern, content_filtered)
    print(f"Extracted content tags: {content_tags}")
    
    # Combine all tags
    all_tags = set(frontmatter_tags + content_tags)
    print(f"Combined tags (before deduplication): {all_tags}")
    
    # Deduplicate tags case-insensitively
    tags_lower_dict = {}
    for tag in all_tags:
        tag_lower = tag.lower()
        print(f"Processing tag: '{tag}' (lower: '{tag_lower}')")
        if tag_lower not in tags_lower_dict:
            tags_lower_dict[tag_lower] = tag
            print(f"  Added as unique")
        else:
            print(f"  Skipped as duplicate of '{tags_lower_dict[tag_lower]}'")
    
    # Return deduplicated tags
    result = list(tags_lower_dict.values())
    print(f"Final deduplicated tags: {result}")
    return result

def extract_and_dedupe_links(content):
    """Extract and deduplicate links from content"""
    # Extract internal links
    internal_link_pattern = r'\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]'
    internal_links = re.findall(internal_link_pattern, content)
    print(f"Extracted raw links: {internal_links}")
    
    # Deduplicate links case-insensitively
    links_lower_dict = {}
    for link in internal_links:
        link_stripped = link.strip()
        link_lower = link_stripped.lower()
        print(f"Processing link: '{link_stripped}' (lower: '{link_lower}')")
        if link_lower not in links_lower_dict:
            links_lower_dict[link_lower] = link_stripped
            print(f"  Added as unique")
        else:
            print(f"  Skipped as duplicate of '{links_lower_dict[link_lower]}'")
    
    # Return deduplicated links
    result = list(links_lower_dict.values())
    print(f"Final deduplicated links: {result}")
    return result

def main():
    # Use the global TEST_CONTENT variable
    content = TEST_CONTENT
    
    print("Testing tag and link deduplication...\n")
    print("Input content:")
    print("-" * 40)
    print(content)
    print("-" * 40 + "\n")
    
    # Extract and deduplicate tags
    print("\n=== TAGS DEDUPLICATION ===")
    deduplicated_tags = extract_and_dedupe_tags(content)
    print("\nDeduplicated tags (final result):")
    print(json.dumps(deduplicated_tags, indent=2))
    print(f"Number of unique tags: {len(deduplicated_tags)}\n")
    
    # Extract and deduplicate links
    print("\n=== LINKS DEDUPLICATION ===")
    deduplicated_links = extract_and_dedupe_links(content)
    print("\nDeduplicated links (final result):")
    print(json.dumps(deduplicated_links, indent=2))
    print(f"Number of unique links: {len(deduplicated_links)}\n")
    
    # Additional test with real content if provided
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            print(f"\n\n=== TESTING WITH REAL CONTENT FROM {file_path} ===\n")
            
            print("\n=== TAGS DEDUPLICATION (REAL FILE) ===")
            deduplicated_tags = extract_and_dedupe_tags(file_content)
            print("\nDeduplicated tags (final result):")
            print(json.dumps(deduplicated_tags, indent=2))
            print(f"Number of unique tags: {len(deduplicated_tags)}")
            
            print("\n=== LINKS DEDUPLICATION (REAL FILE) ===")
            deduplicated_links = extract_and_dedupe_links(file_content)
            print("\nDeduplicated links (final result):")
            print(json.dumps(deduplicated_links, indent=2))
            print(f"Number of unique links: {len(deduplicated_links)}")

if __name__ == "__main__":
    main() 