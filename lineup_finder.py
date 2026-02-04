"""
Lineup Finder - AI-Powered Event Artist Extraction

Analyzes event descriptions using OpenAI GPT-4 to identify performing artists,
enriches data with Spotify metrics, and generates prioritized outreach lists.

Built at Eventbrite to support the Lineup feature launch.
Shared with permission from Eventbrite legal and engineering teams.
"""

import requests
import json
import sys
import os
import csv
import time
import base64
import re 
from collections import defaultdict
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    print("Error: 'tqdm' library not found. Please install it using: pip install tqdm")
    sys.exit(1)

# --- CONFIGURATION ---
# Replace these with your actual API credentials
EVENTBRITE_API_KEY = "YOUR_EVENTBRITE_API_KEY_GOES_HERE"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_GOES_HERE"
SPOTIFY_CLIENT_ID = "YOUR_SPOTIFY_CLIENT_ID_GOES_HERE"
SPOTIFY_CLIENT_SECRET = "YOUR_SPOTIFY_CLIENT_SECRET_GOES_HERE"

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ERROR_LOG_FILE = "error_log.txt"


def log_error_to_file(event_id: str, text_sent: str, raw_response: str, reason: str = "Parsing Error"):
    """
    Appends detailed error information to a log file for debugging.
    
    Args:
        event_id: The Eventbrite event ID
        text_sent: The text sent to the AI model
        raw_response: The raw response from the AI
        reason: Brief description of the error
    """
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Event ID: {event_id}\n")
            f.write(f"Reason: {reason}\n")
            f.write("\n--- Text Sent to AI ---\n")
            f.write(text_sent + "\n")
            f.write("\n--- Raw AI Response ---\n")
            f.write(raw_response + "\n")
            f.write("="*80 + "\n\n")
    except IOError as e:
        tqdm.write(f"\n[!] Critical: Could not write to error log file: {e}")


def get_spotify_access_token() -> str | None:
    """
    Obtains a Spotify API access token using client credentials flow.
    
    Returns:
        Access token string, or None if authentication fails
    """
    auth_url = "https://accounts.spotify.com/api/token"
    auth_header_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_header_base64 = base64.b64encode(auth_header_string.encode('utf-8'))
    headers = {"Authorization": f"Basic {auth_header_base64.decode('utf-8')}"}
    data = {"grant_type": "client_credentials"}
    
    try:
        response = requests.post(auth_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"\n[!] Critical Error: Could not get Spotify access token. {e}")
        return None


def clean_name(name: str) -> str:
    """
    Normalizes an artist name for fuzzy matching by removing punctuation,
    whitespace, and common stop words.
    
    Examples:
        "The Rolling Stones" → "rollingstones"
        "Rage Against The Machine" → "rageagainstmachine"
        "deadmau5" → "deadmau5" (preserves numbers)
    
    Args:
        name: Artist name to clean
        
    Returns:
        Cleaned name string (lowercase, no punctuation/spaces)
    """
    if not name:
        return ""
    
    # Words to ignore during matching
    stop_words = ['the', 'band', 'djs', 'ft', 'feat', 'and', '&']
    
    # Convert to lowercase
    cleaned = name.lower()
    
    # Remove punctuation
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    
    # Remove stop words and collapse whitespace
    cleaned = "".join([word for word in cleaned.split() if word not in stop_words])
    
    return cleaned


def add_spotify_data(artists: list[dict], access_token: str):
    """
    Enriches a list of artist dictionaries with Spotify data using multi-step
    matching logic (exact match → cleaned name match).
    
    Modifies artist dictionaries in-place to add:
        - spotifyVerified: bool (whether a match was found)
        - spotifyUrl: str (Spotify artist URL)
        - spotifyFollowers: int (follower count)
        - spotifyMatch: str ("Perfect", "Partial", or None)
    
    Args:
        artists: List of artist dictionaries with 'artistName' keys
        access_token: Valid Spotify API access token
    """
    if not access_token or not artists:
        return
    
    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    for artist_info in artists:
        # Initialize Spotify fields
        artist_info['spotifyVerified'] = False
        artist_info['spotifyUrl'] = None
        artist_info['spotifyFollowers'] = 0
        artist_info['spotifyMatch'] = None 
        
        name = artist_info.get('artistName')
        if not name: 
            continue

        params = {"q": name, "type": "artist", "limit": 1}
        
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                tqdm.write("  [!] Spotify rate limit hit. Pausing...")
                time.sleep(10)
                response = requests.get(search_url, headers=headers, params=params, timeout=10)

            response.raise_for_status()
            search_results = response.json()
            items = search_results.get("artists", {}).get("items", [])
            
            if items:
                top_artist = items[0]
                top_artist_name = top_artist.get('name', '')
                
                # Step 1: Check for exact match (case-insensitive)
                if name.lower() == top_artist_name.lower():
                    artist_info['spotifyVerified'] = True
                    artist_info['spotifyMatch'] = "Perfect"
                
                # Step 2: Check for cleaned name match (fuzzy)
                else:
                    clean_query_name = clean_name(name)
                    clean_result_name = clean_name(top_artist_name)

                    if clean_query_name and clean_query_name == clean_result_name:
                        artist_info['spotifyVerified'] = True
                        artist_info['spotifyMatch'] = "Partial"
                
                # If match found, update Spotify data
                if artist_info['spotifyVerified']:
                    artist_info['spotifyUrl'] = top_artist.get("external_urls", {}).get("spotify")
                    artist_info['spotifyFollowers'] = top_artist.get("followers", {}).get("total", 0)

        except requests.exceptions.RequestException:
            # Silently continue on API errors
            pass


def get_artists_from_text(event_id: str, event_text: str) -> list[dict]:
    """
    Uses OpenAI GPT-4 to extract performing artist information from event text.
    
    Implements retry logic for token limit errors and comprehensive error logging.
    
    Args:
        event_id: Event ID for logging purposes
        event_text: Combined event title and description
        
    Returns:
        List of artist dictionaries with keys:
            - artistName: str
            - confidence: str ("high", "medium", "low")
            - isTribute: bool
            - reasoning: str (AI explanation)
    """
    if not event_text or not event_text.strip(): 
        return []
    
    # Carefully engineered prompt for accurate artist extraction
    system_prompt = """You are a highly critical event analysis expert. Your task is to extract every person or group mentioned in the text and then determine if they are a performer at the current event.

**Critical Analysis Steps:**
1.  **Extract All Names:** First, identify every potential proper name that could be a person or a performing group.
2.  **Determine if Tribute:** For each name, check for keywords like "tribute", "cover band", "experience", "celebration of".
    -   If an artist is a tribute band (e.g., "Who's Bad: The Ultimate Michael Jackson Experience"), set a boolean flag `isTribute` to `true`.
    -   If it is not a tribute, set `isTribute` to `false`.
    -   If the original artist is mentioned in the context of a tribute (e.g., Michael Jackson), they are NOT performing. Their confidence must be "low" and `isTribute` should be `false` for them.
3.  **Determine Confidence:** Assign a confidence level (`high`, `medium`, `low`) for whether the named entity is performing at the event.
    -   `high`: Explicitly stated as performing (e.g., "featuring...", "live performance by..."). A tribute band that is performing gets `high` confidence.
    -   `medium`: Ambiguous context, could be a host or other participant.
    -   `low`: Clearly not performing (e.g., a past guest, or the original artist in a tribute).
4.  **Handle Transportation:** If the event is a "bus", "shuttle", "ride to" a concert, any mentioned artists are not performing at THIS event. Return an empty list.

**Output Format:**
Your final output MUST be a single, valid JSON object with a key named `"lineup"` which contains a JSON array of artist objects. Each object must have "artistName", "confidence" (string), "isTribute" (boolean), and a very verbose "reasoning".
**Example Output Structure:** `{"lineup": [{"artistName": "Who's Bad", "confidence": "high", "isTribute": true, "reasoning": "Identified as the main act and a tribute band."}]}`
---
Now, analyze the following event details using the same critical steps. Be strict.
"""
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}", 
        "Content-Type": "application/json"
    }
    
    current_max_tokens = 1500
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": event_text}
        ],
        "max_tokens": current_max_tokens, 
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    
    raw_response_text = ""
    finish_reason = None
    
    # First attempt
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code >= 500: 
            response.raise_for_status() 
        
        result = response.json()
        raw_response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish_reason = result.get("choices", [{}])[0].get("finish_reason", None)
        
        # If response wasn't truncated, parse and return
        if finish_reason != "length":
            response_data = json.loads(raw_response_text)
            return response_data.get("lineup", [])
            
    except requests.exceptions.RequestException as e:
        tqdm.write(f"\n[!] OpenAI API error on first attempt for event {event_id}: {e}")
    except (json.JSONDecodeError, IndexError, KeyError):
         tqdm.write(f"\n[!] Failed to parse OpenAI response on first attempt for event {event_id}.")
         log_error_to_file(event_id, event_text, raw_response_text, reason="Parsing Error (Attempt 1)")
         return [] 

    # Retry with increased token limit if truncated
    if finish_reason == "length":
        tqdm.write(f"\n[!] OpenAI response likely truncated for event {event_id}. Retrying with more tokens...")
        current_max_tokens = 3000 
        payload["max_tokens"] = current_max_tokens
        
        try:
            response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=90)
            if response.status_code >= 500: 
                response.raise_for_status()

            result = response.json()
            raw_response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            finish_reason_retry = result.get("choices", [{}])[0].get("finish_reason", None)

            response_data = json.loads(raw_response_text)
            
            if finish_reason_retry == "length":
                 tqdm.write(f"\n[!] Warning: OpenAI response still truncated after retry for event {event_id}.")
                 log_error_to_file(event_id, event_text, raw_response_text, reason="Truncated (Retry)")

            return response_data.get("lineup", [])

        except requests.exceptions.RequestException as e:
            tqdm.write(f"\n[!] OpenAI API error on retry attempt for event {event_id}: {e}")
            log_error_to_file(event_id, event_text, f"Retry failed with RequestException: {e}", reason="API Error (Retry)")
        except (json.JSONDecodeError, IndexError, KeyError):
             tqdm.write(f"\n[!] Failed to parse OpenAI response on retry attempt for event {event_id}.")
             log_error_to_file(event_id, event_text, raw_response_text, reason="Parsing Error (Retry)")
             
    return []


def main():
    """
    Main execution function. Processes events from CSV, identifies lineups,
    enriches with Spotify data, and generates prioritized output file.
    """
    if len(sys.argv) < 3:
        print("Usage: python3 lineup_finder.py <input_file.csv> <events_per_creator>")
        print("       (use 0 for events_per_creator to include all events)")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    output_file_path = "lineup_suggestions.csv"
    
    try:
        limit_per_creator = int(sys.argv[2])
    except ValueError:
        print("[!] Error: <events_per_creator> must be an integer (e.g., 0, 1, 2).")
        sys.exit(1)

    if not os.path.exists(input_file_path):
        print(f"[!] Error: The input file '{input_file_path}' was not found.")
        sys.exit(1)

    print("[*] Acquiring Spotify Access Token...")
    spotify_token = get_spotify_access_token()
    if not spotify_token: 
        sys.exit(1)

    print(f"[*] Reading and grouping events from '{input_file_path}'...")
    unique_events = defaultdict(lambda: {"metadata": None, "html_texts": []})
    
    try:
        with open(input_file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            required_columns = [
                "EVENT_ID", "EVENT_TITLE", "HTML_TEXT", "CREATOR_NAME", 
                "EVENT_START_DATE", "COUNTRY_NAME", "CAPACITY"
            ]
            
            if not all(col in reader.fieldnames for col in required_columns):
                missing = [col for col in required_columns if col not in reader.fieldnames]
                print(f"[!] Error: Input CSV is missing required columns: {', '.join(missing)}")
                sys.exit(1)

            for row in reader:
                event_id = row["EVENT_ID"]
                if not unique_events[event_id]["metadata"]:
                    unique_events[event_id]["metadata"] = row
                if row.get("HTML_TEXT"):
                    unique_events[event_id]["html_texts"].append(row["HTML_TEXT"])
    
    except Exception as e:
        print(f"[!] Error reading or grouping CSV file: {e}")
        sys.exit(1)

    if not unique_events:
        print("[-] No valid events found in the input file.")
        return
    
    output_headers = [
        "EXTENAL_ID", "CREATOR_NAME", "CREATOR_EMAIL", "TRANSACTED_GTF_USD_12MO", 
        "EVENT_ID", "EVENT_TITLE", "COUNTRY_NAME", "EVENT_START_DATE", "CAPACITY", 
        "MULTIPLE ARTISTS IDENTIFIED (Y/N)",
        "ARTIST_1_NAME_SPOTIFY", "ARTIST_2_NAME_SPOTIFY", "ARTIST_3_NAME_SPOTIFY",
        "ARTIST_1_NAME_SPOTIFY_LINK", "ARTIST_2_NAME_SPOTIFY_LINK", "ARTIST_3_NAME_SPOTIFY_LINK",
        "ARTIST_1_NAME_SPOTIFY_FOLLOWERS", "ARTIST_2_NAME_SPOTIFY_FOLLOWERS", "ARTIST_3_NAME_SPOTIFY_FOLLOWERS",
        "TOTAL_SPOTIFY_FOLLOWERS",
        "REVIEW_FLAGS"
    ]

    print(f"[*] Processing {len(unique_events)} unique events...")
    all_found_lineups = []
    lineups_found_count = 0 

    event_iterator = tqdm(
        unique_events.items(), 
        total=len(unique_events), 
        desc="Finding Lineups", 
        unit="event"
    )

    for event_id, data in event_iterator:
        metadata = data["metadata"]
        title = metadata.get("EVENT_TITLE", "")
        full_html = "\n".join(data["html_texts"])
        full_text_for_analysis = ' '.join(f"{title}\n{full_html}".split()).strip() 

        if not full_text_for_analysis:
             continue

        # Get artists from AI
        ai_artists = get_artists_from_text(event_id, full_text_for_analysis)
        if not ai_artists: 
            continue
        
        # Enrich with Spotify data
        add_spotify_data(ai_artists, spotify_token)
        
        # Filter to high-confidence, non-tribute performers
        suggested_lineup = [
            artist for artist in ai_artists
            if artist.get('confidence', 'low') == 'high' 
            and not artist.get('isTribute')
        ]
        
        if suggested_lineup:
            # Sort by Spotify followers (descending)
            sorted_artists = sorted(
                suggested_lineup, 
                key=lambda x: x.get('spotifyFollowers', 0), 
                reverse=True
            )
            
            # Minimum follower filter (reduces noise)
            if sorted_artists[0].get('spotifyFollowers', 0) < 100:
                continue 

            lineups_found_count += 1
            total_followers = sum(a.get('spotifyFollowers', 0) for a in sorted_artists)
            
            # Generate review flags for quality assurance
            review_flags = []
            artist_1_followers = sorted_artists[0].get('spotifyFollowers', 0)

            # Rule 1: Follower/Capacity Ratio (likely mismatch)
            try:
                capacity_str = metadata.get("CAPACITY", "0")
                capacity = int(float(capacity_str)) if capacity_str else 0
                if capacity > 0 and (artist_1_followers / capacity) > 25000:
                    review_flags.append("Follower/Capacity Ratio (>25000)")
            except (ValueError, TypeError, ZeroDivisionError):
                pass 
            
            # Rule 2: Follower disparity between artists
            if len(sorted_artists) >= 2:
                artist_2_followers = sorted_artists[1].get('spotifyFollowers', 0)
                if artist_2_followers > 0 and (artist_1_followers / artist_2_followers) > 1000:
                    review_flags.append("Follower Ratio (>1000x)")

            # Rules 3-5: Partial Spotify matches (fuzzy matching used)
            if len(sorted_artists) > 0 and sorted_artists[0].get('spotifyMatch') == "Partial":
                review_flags.append("Artist 1 Partial Match")
            if len(sorted_artists) > 1 and sorted_artists[1].get('spotifyMatch') == "Partial":
                review_flags.append("Artist 2 Partial Match")
            if len(sorted_artists) > 2 and sorted_artists[2].get('spotifyMatch') == "Partial":
                review_flags.append("Artist 3 Partial Match")

            all_found_lineups.append({
                "creator_name": metadata.get("CREATOR_NAME"),
                "total_followers": total_followers,
                "start_date": metadata.get("EVENT_START_DATE"),
                "data_row": {
                    "EXTENAL_ID": metadata.get("EXTERNAL_ID"),
                    "CREATOR_NAME": metadata.get("CREATOR_NAME"),
                    "CREATOR_EMAIL": metadata.get("CREATOR_EMAIL"),
                    "TRANSACTED_GTF_USD_12MO": metadata.get("TRANSACTED_GTF_USD_12MO"),
                    "EVENT_ID": event_id,
                    "EVENT_TITLE": metadata.get("EVENT_TITLE"), 
                    "COUNTRY_NAME": metadata.get("COUNTRY_NAME"), 
                    "EVENT_START_DATE": metadata.get("EVENT_START_DATE"),
                    "CAPACITY": metadata.get("CAPACITY"),
                    "MULTIPLE ARTISTS IDENTIFIED (Y/N)": "Y" if len(sorted_artists) > 1 else "N",
                    "TOTAL_SPOTIFY_FOLLOWERS": total_followers,
                    
                    "ARTIST_1_NAME_SPOTIFY": sorted_artists[0].get('artistName', '') if len(sorted_artists) > 0 else '',
                    "ARTIST_2_NAME_SPOTIFY": sorted_artists[1].get('artistName', '') if len(sorted_artists) > 1 else '',
                    "ARTIST_3_NAME_SPOTIFY": sorted_artists[2].get('artistName', '') if len(sorted_artists) > 2 else '',
                    
                    "ARTIST_1_NAME_SPOTIFY_LINK": sorted_artists[0].get('spotifyUrl', '') if len(sorted_artists) > 0 else '',
                    "ARTIST_2_NAME_SPOTIFY_LINK": sorted_artists[1].get('spotifyUrl', '') if len(sorted_artists) > 1 else '',
                    "ARTIST_3_NAME_SPOTIFY_LINK": sorted_artists[2].get('spotifyUrl', '') if len(sorted_artists) > 2 else '',
                    
                    "ARTIST_1_NAME_SPOTIFY_FOLLOWERS": sorted_artists[0].get('spotifyFollowers', '') if len(sorted_artists) > 0 else '',
                    "ARTIST_2_NAME_SPOTIFY_FOLLOWERS": sorted_artists[1].get('spotifyFollowers', '') if len(sorted_artists) > 1 else '',
                    "ARTIST_3_NAME_SPOTIFY_FOLLOWERS": sorted_artists[2].get('spotifyFollowers', '') if len(sorted_artists) > 2 else '',
                    
                    "REVIEW_FLAGS": "; ".join(review_flags)
                }
            })
            
        event_iterator.set_postfix_str(f"Found: {lineups_found_count}")

    print(f"[*] Found {lineups_found_count} valid lineups. Filtering by creator limit...")
    final_rows_to_write = []
    
    if limit_per_creator == 0:
        # Include all events
        final_rows_to_write = [result["data_row"] for result in all_found_lineups]
    else:
        # Limit events per creator, sorted by follower count and date
        events_by_creator = defaultdict(list)
        for result in all_found_lineups:
            creator_key = result["creator_name"] if result["creator_name"] else "Unknown Creator"
            events_by_creator[creator_key].append(result)

        for creator, events in events_by_creator.items():
            try:
                events.sort(key=lambda x: (
                    -x["total_followers"], 
                    datetime.strptime(x["start_date"], "%Y-%m-%d %H:%M:%S.%f")
                ))
            except (ValueError, TypeError):
                 try:
                    events.sort(key=lambda x: (
                        -x["total_followers"], 
                        datetime.strptime(x["start_date"], "%Y-%m-%d %H:%M:%S")
                    ))
                 except (ValueError, TypeError):
                    events.sort(key=lambda x: -x["total_followers"])
                    tqdm.write(f"\n[!] Warning: Could not parse start date for creator '{creator}'. Using followers only.")

            for result in events[:limit_per_creator]:
                final_rows_to_write.append(result["data_row"])

    print(f"[*] Sorting {len(final_rows_to_write)} final events by main artist followers...")
    try:
        final_rows_to_write.sort(
            key=lambda x: int(x.get('ARTIST_1_NAME_SPOTIFY_FOLLOWERS') or 0), 
            reverse=True
        )
    except Exception as e:
        print(f"\n[!] Warning: Could not sort final list by followers. Error: {e}")

    # Write output CSV
    with open(output_file_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(final_rows_to_write)
    
    print("\n" + "="*40)
    print(" " * 11 + "PROCESS COMPLETE")
    print("="*40)
    print(f"Found potential lineups in {lineups_found_count} events (after filtering).")
    print(f"Wrote {len(final_rows_to_write)} final events to '{output_file_path}'.")
    print(f"[✔] Enriched data saved.")


if __name__ == "__main__":
    main()
