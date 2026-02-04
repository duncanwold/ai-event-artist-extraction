# Lineup Finder - AI-Powered Event Artist Extraction

> Intelligent event analysis tool that uses LLMs to identify performing artists and enrich event data with Spotify metrics

## Overview

Lineup Finder is a production data enrichment tool built at Eventbrite that analyzes thousands of event descriptions using AI to automatically identify performing artists, verify them against Spotify, and generate actionable insights for targeted marketing campaigns.

The tool powered successful email campaigns to event creators, suggesting they add artist lineup information to their events, resulting in **above-benchmark performance** across all key metrics (CTR, open rate, adoption rate).

## The Problem

Eventbrite launched a "Lineup" feature that allows creators to tag performing artists, which then:
- Distributes events to **Spotify** (automatic tour date listings)
- Syncs with **Songkick** for concert discovery
- Enhances **Google Search** results with rich event cards
- Integrates with **Bandsintown** for fan notifications

**Challenge:** Thousands of music events didn't use this feature because creators didn't know it existed or understand its value.

**Solution:** Proactively identify which events *should* have lineup data, measure the artists' reach (via Spotify followers), and target outreach to creators with the highest-value opportunities.

## How It Works

### 1. **Intelligent Artist Extraction (OpenAI GPT-4)**

The tool uses a carefully engineered prompt to analyze event titles and descriptions:

```
Event: "Who's Bad: The Ultimate Michael Jackson Experience"
```

**AI Analysis:**
- ✅ Identifies "Who's Bad" as the performing artist
- ✅ Recognizes it's a tribute band (`isTribute: true`)
- ❌ Correctly excludes "Michael Jackson" (not performing, despite being mentioned)
- ✅ Assigns confidence level (`high`, `medium`, `low`)

**Context-Aware Logic:**
- Detects tribute bands vs original artists
- Filters out transportation events ("bus to see [Artist]")
- Distinguishes performers from hosts, past guests, or referenced artists

### 2. **Spotify Data Enrichment**

For each identified artist:
- **Searches Spotify API** for matching artist profiles
- **Multi-step matching logic:**
  1. Exact name match (case-insensitive)
  2. "Cleaned" name match (removes punctuation, stop words, spaces)
  3. Verification flag for partial matches
- **Extracts metrics:**
  - Spotify URL
  - Follower count
  - Verification status

### 3. **Data Quality & Prioritization**

**Filtering Rules:**
- Minimum 100 Spotify followers (filters noise)
- High-confidence performers only
- Non-tribute acts prioritized

**Review Flags for Edge Cases:**
- Follower/Capacity ratio >25,000x (likely mismatch)
- Artist follower disparity >1,000x (potential error)
- Partial Spotify matches (needs verification)

### 4. **Smart Output Ranking**

Events sorted by:
1. **Total Spotify followers** (proxy for marketing impact)
2. **Event start date** (prioritize upcoming events)
3. **Per-creator limits** (avoid overwhelming single creators)

## Business Impact

### Marketing Campaign Results
- Enabled **targeted outreach** to creators with high-value lineup opportunities
- Campaign performance **exceeded all benchmarks**:
  - Click-through rate (CTR): Above target
  - Open rate: Above target  
  - Feature adoption rate: Above target
- Drove adoption of Lineup feature, increasing event distribution to Spotify, Google, Songkick, and Bandsintown

### Operational Efficiency
- **Automated analysis** of thousands of events
- **Prioritized outreach** based on artist reach (Spotify followers)
- **Reduced manual review** through confidence scoring and validation rules

## Technical Architecture

### APIs Integrated
- **OpenAI GPT-4** - Natural language understanding for artist extraction
- **Spotify Web API** - Artist verification and metrics
- **Eventbrite API** - Event data source (input CSV)

### Key Features

**Robust Error Handling:**
- Retry logic for API rate limits
- Token limit detection and expansion (1500 → 3000 tokens)
- Comprehensive error logging with full context
- Graceful degradation when APIs fail

**Performance Optimizations:**
- Progress bars (tqdm) for long-running processes
- Batch processing with configurable limits
- Efficient data grouping (multiple HTML blocks per event)

**Data Validation:**
- Confidence scoring (`high`, `medium`, `low`)
- Tribute band detection
- Context-aware filtering (transportation events)
- Review flags for manual QA

## Usage

### Installation
```bash
pip install requests tqdm
```

### Configuration
Update API credentials in the script:
```python
EVENTBRITE_API_KEY = "your_key"
OPENAI_API_KEY = "your_key"  
SPOTIFY_CLIENT_ID = "your_id"
SPOTIFY_CLIENT_SECRET = "your_secret"
```

### Running the Tool
```bash
python lineup_finder.py events.csv 2
```

**Arguments:**
- `events.csv` - Input file with event data
- `2` - Maximum events per creator (use `0` for unlimited)

### Input CSV Format
Required columns:
```
EVENT_ID, EVENT_TITLE, HTML_TEXT, CREATOR_NAME, 
EVENT_START_DATE, COUNTRY_NAME, CAPACITY
```

### Output
`lineup_suggestions.csv` with:
- Up to 3 artists per event (sorted by Spotify followers)
- Spotify URLs and follower counts
- Total follower count
- Review flags for manual QA
- Sorted by artist reach (highest followers first)

## Example Output

| Event Title | Artist 1 | Followers | Artist 2 | Followers | Total Followers | Review Flags |
|------------|----------|-----------|----------|-----------|-----------------|--------------|
| Summer Music Fest | Taylor Swift | 93M | Ed Sheeran | 82M | 175M | - |
| Tribute Night | Journey Revisited | 1.2K | - | - | 1.2K | Artist 1 Partial Match |

## Technical Highlights

### Prompt Engineering
The OpenAI prompt uses:
- **Chain-of-thought reasoning** (step-by-step analysis)
- **Strict JSON formatting** enforcement
- **Multi-class confidence** scoring
- **Context-aware filtering** rules

### Name Matching Algorithm
```python
def clean_name(name: str) -> str:
    # 1. Lowercase normalization
    # 2. Punctuation removal  
    # 3. Stop word filtering ("the", "band", "ft", etc.)
    # 4. Whitespace collapse
```

Handles cases like:
- "The Rolling Stones" → "rollingstones"
- "deadmau5" → "deadmau5" (preserves numbers)
- "Rage Against The Machine" → "rageagainstmachine"

### Error Recovery
- **Token limit retry:** Automatically increases from 1500 to 3000 tokens
- **Rate limit handling:** 10-second pause on Spotify 429 errors
- **Detailed logging:** Full request/response captured for debugging

## Code Quality Features

✅ **Type hints** for function signatures  
✅ **Comprehensive error handling**  
✅ **Progress indicators** for user feedback  
✅ **Modular design** (separate functions for each API)  
✅ **Configuration management** (API keys centralized)  
✅ **Logging infrastructure** (error_log.txt)  

## Lessons Learned

**🤖 LLM Prompt Engineering is Critical**
Initial versions struggled with tribute bands and transportation events. Iterative prompt refinement with explicit reasoning steps dramatically improved accuracy.

**📊 Data Quality > Data Quantity**
Adding validation rules and review flags was essential. Better to flag 10% for manual review than spam creators with false positives.

**🎯 Follower Count Proxy Works**
Spotify followers proved to be an excellent proxy for marketing impact. Events featuring artists with 1M+ followers had significantly higher campaign engagement.

**⚡ Rate Limits are Real**
Both OpenAI and Spotify have rate limits. Built-in retry logic and progress tracking made the tool production-ready.

## Future Enhancements

- [ ] Add Songkick API for additional artist verification
- [ ] Machine learning model to predict lineup adoption likelihood
- [ ] Real-time processing via event webhooks
- [ ] Support for non-music events (comedians, speakers, etc.)
- [ ] Multi-language event description support
- [ ] Confidence calibration based on historical accuracy

## Technology Stack

- **Python 3.10+**
- **OpenAI GPT-4** via API
- **Spotify Web API**
- **Libraries:** requests, tqdm

## Attribution

Built at **Eventbrite** to support the Lineup feature launch. This implementation is shared with permission from Eventbrite's legal and engineering teams. The Lineup feature concept and business strategy are owned by Eventbrite.

## Background

This project was developed as part of Eventbrite's efforts to enhance event discovery and distribution. The tool exemplifies practical application of:
- Large Language Models for entity extraction
- Multi-API data enrichment
- Data-driven marketing prioritization
- Production-ready error handling

---

*"Vibe-coded" with assistance from Gemini, refined through production use*
