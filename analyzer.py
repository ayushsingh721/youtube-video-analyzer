"""
analyzer.py - Core NLP Analysis Engine
Handles YouTube API calls + all NLP processing
"""

import re
import string
from collections import Counter

import nltk
import requests
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ─── Download required NLTK data (runs once) ──────────────────────────────────
for pkg in ["punkt", "stopwords", "vader_lexicon", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


class VideoAnalyzer:
    """
    Full pipeline: YouTube API fetch → NLP processing → structured results.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words("english"))
        # Add YouTube-specific noise words
        self.stop_words.update([
            "video", "watch", "like", "subscribe", "channel", "youtube",
            "comment", "share", "click", "link", "check", "please", "also",
            "would", "could", "one", "us", "get", "use", "make", "know",
            "https", "http", "www", "com"
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # YouTube API Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_video_details(self, video_id: str) -> dict:
        """Fetch video metadata from YouTube Data API v3."""
        url = f"{YOUTUBE_API_BASE}/videos"
        params = {
            "part": "snippet,statistics",
            "id": video_id,
            "key": self.api_key,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("items"):
            return {}

        item = data["items"][0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        return {
            "video_id": video_id,
            "title": snippet.get("title", "N/A"),
            "channel": snippet.get("channelTitle", "N/A"),
            "description": snippet.get("description", ""),
            "tags": snippet.get("tags", []),
            "published_at": snippet.get("publishedAt", "")[:10],
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
        }

    def _fetch_top_comments(self, video_id: str, max_results: int = 50) -> list[str]:
        """Fetch top comments from the video."""
        url = f"{YOUTUBE_API_BASE}/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": max_results,
            "order": "relevance",
            "key": self.api_key,
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            items = response.json().get("items", [])
            return [
                item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
                for item in items
            ]
        except Exception:
            return []  # Comments may be disabled on some videos

    # ──────────────────────────────────────────────────────────────────────────
    # NLP Processing
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Remove URLs, special characters, and normalize whitespace."""
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def _tokenize_and_filter(self, text: str) -> list[str]:
        """Tokenize text and remove stopwords + short tokens."""
        cleaned = self._clean_text(text)
        tokens = word_tokenize(cleaned)
        return [
            t for t in tokens
            if t.isalpha() and t not in self.stop_words and len(t) > 2
        ]

    def _get_word_frequency(self, tokens: list[str], top_n: int = 15) -> list[dict]:
        """Return top-N word frequency pairs."""
        freq = Counter(tokens)
        return [{"word": w, "count": c} for w, c in freq.most_common(top_n)]

    def _extract_keywords(self, tokens: list[str], top_n: int = 10) -> list[str]:
        """Extract keywords using TF-style frequency filtering."""
        freq = Counter(tokens)
        total = sum(freq.values()) or 1
        # Keywords = tokens appearing more than average frequency
        avg = total / len(freq) if freq else 0
        keywords = [w for w, c in freq.most_common(30) if c > avg]
        return keywords[:top_n]

    def _analyze_sentiment(self, texts: list[str]) -> dict:
        """
        Run VADER sentiment on a list of text strings.
        Returns counts + per-comment scores for the top 20 comments.
        """
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        sample = []

        for text in texts:
            scores = self.sia.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            counts[label] += 1
            if len(sample) < 20:
                sample.append({
                    "text": text[:200],
                    "label": label,
                    "score": round(compound, 3),
                })

        total = sum(counts.values()) or 1
        return {
            "counts": counts,
            "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
            "sample_comments": sample,
        }

    def _generate_summary(self, text: str, num_sentences: int = 4) -> str:
        """
        Extractive summarization: score sentences by keyword density,
        pick the top `num_sentences`.
        """
        if not text or len(text) < 100:
            return text

        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        tokens = self._tokenize_and_filter(text)
        freq = Counter(tokens)
        max_freq = max(freq.values()) if freq else 1

        # Score each sentence by normalized word frequency
        scores = {}
        for sent in sentences:
            sent_tokens = self._tokenize_and_filter(sent)
            scores[sent] = sum(freq.get(t, 0) / max_freq for t in sent_tokens)

        top_sentences = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
        # Return in original order
        return " ".join(s for s in sentences if s in top_sentences)

    # ──────────────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, video_id: str) -> dict:
        """
        Full pipeline:  API fetch → NLP → structured JSON result.
        """
        # 1. Fetch video metadata
        details = self._fetch_video_details(video_id)
        if not details:
            return {"error": "Video not found. Check the URL or Video ID."}

        # 2. Fetch comments
        comments = self._fetch_top_comments(video_id)

        # 3. Build combined text corpus (description + tags + comments)
        description = details.get("description", "")
        tags_text = " ".join(details.get("tags", []))
        comments_combined = " ".join(comments)
        full_corpus = f"{description} {tags_text} {comments_combined}"

        # 4. NLP on corpus
        all_tokens = self._tokenize_and_filter(full_corpus)
        desc_tokens = self._tokenize_and_filter(description)

        word_freq = self._get_word_frequency(all_tokens, top_n=15)
        keywords = self._extract_keywords(all_tokens, top_n=10)
        sentiment = self._analyze_sentiment(comments) if comments else {
            "counts": {"positive": 0, "negative": 0, "neutral": 0},
            "percentages": {"positive": 0, "negative": 0, "neutral": 0},
            "sample_comments": [],
        }
        summary = self._generate_summary(
            description if len(description) > 200 else comments_combined
        )

        # 5. Format numbers for display
        from utils import format_number
        return {
            "video_id": video_id,
            "thumbnail": details["thumbnail"],
            "title": details["title"],
            "channel": details["channel"],
            "published_at": details["published_at"],
            "description": description[:600] + ("..." if len(description) > 600 else ""),
            "tags": details["tags"][:15],
            "stats": {
                "views": format_number(details["view_count"]),
                "likes": format_number(details["like_count"]),
                "comments": format_number(details["comment_count"]),
                "views_raw": details["view_count"],
                "likes_raw": details["like_count"],
            },
            "nlp": {
                "word_frequency": word_freq,
                "keywords": keywords,
                "sentiment": sentiment,
                "summary": summary,
                "total_tokens": len(all_tokens),
                "unique_tokens": len(set(all_tokens)),
            },
            "comments_fetched": len(comments),
        }
