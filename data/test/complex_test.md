---
title: Complex Deduplication Test
tags: 
  - firstTag
  - secondTag
  - ThirdTag
  - thirdtag
  - energy
---

# Complex Deduplication Test

This file tests various deduplication scenarios.

## Duplicate Internal Links
Here is a link: [[energy]]
Here is the same link again: [[energy]]
Here is the same link with different case: [[Energy]]
Here is a link with an alias: [[energy|Energy Source]]

## Duplicate Tags
Tags in frontmatter and content should deduplicate:
#firstTag #secondTag #fourthTag #fifthTag #fifthTag

## Tags that match Internal Links
Both tag and link: #energy and [[energy]] should only appear in the tags list.

## Formatting Edge Cases
Code with hashtag: `#not-a-tag`
URL with hashtag: https://example.com/#not-a-tag
