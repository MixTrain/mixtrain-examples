"""Publish a weekday digest of recent arXiv papers."""

from datetime import datetime, timedelta, timezone
from typing import Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from mixtrain import Markdown, MixRoutine, on_schedule


ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}

Category = Literal[
    "Machine Learning",
    "Artificial Intelligence",
    "Computation and Language",
    "Computer Vision and Pattern Recognition",
    "Robotics",
    "Neural and Evolutionary Computing",
    "Information Retrieval",
    "Human-Computer Interaction",
    "Distributed, Parallel, and Cluster Computing",
    "Software Engineering",
    "Cryptography and Security",
    "Data Structures and Algorithms",
]


def fetch_papers(query: str, paper_count: int) -> list[dict[str, str]]:
    """Fetch the newest papers matching an arXiv API query."""
    params = urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": paper_count,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    request = Request(
        f"{ARXIV_API}?{params}",
        headers={"User-Agent": "mixtrain-daily-arxiv-digest/1.0"},
    )

    with urlopen(request, timeout=30) as response:
        root = ElementTree.parse(response).getroot()

    papers = []
    for entry in root.findall("atom:entry", ATOM):
        papers.append(
            {
                "title": _text(entry, "atom:title"),
                "url": _text(entry, "atom:id").replace("http://", "https://", 1),
                "published": _text(entry, "atom:published")[:10],
                "authors": ", ".join(
                    _text(author, "atom:name")
                    for author in entry.findall("atom:author", ATOM)
                ),
                "abstract": _text(entry, "atom:summary"),
            }
        )
    return papers


def _text(element: ElementTree.Element, path: str) -> str:
    value = element.findtext(path, default="", namespaces=ATOM)
    return " ".join(value.split())


def _shorten(text: str, word_count: int) -> str:
    words = text.split()
    suffix = "…" if len(words) > word_count else ""
    return " ".join(words[:word_count]) + suffix


class DailyArxivDigest(MixRoutine):
    """Create a Markdown digest from the latest papers in an arXiv search."""

    def run(
        self,
        trigger=on_schedule(cron="0 9 * * 1-5", tz="UTC"),
        category: Category = "Machine Learning",
        keywords: str = "",
        author: str = "",
        paper_count: int = 5,
        abstract_words: int = 60,
    ) -> Markdown:
        """Create a digest using filters that are combined with AND.

        Args:
            category: arXiv subject category.
            keywords: Optional words to find in any paper field.
            author: Optional author name.
            paper_count: Maximum number of papers to include.
            abstract_words: Maximum abstract length per paper.
        """
        if not 1 <= paper_count <= 20:
            raise ValueError("paper_count must be between 1 and 20")
        if not 20 <= abstract_words <= 200:
            raise ValueError("abstract_words must be between 20 and 200")

        category_codes = {
            "Machine Learning": "cs.LG",
            "Artificial Intelligence": "cs.AI",
            "Computation and Language": "cs.CL",
            "Computer Vision and Pattern Recognition": "cs.CV",
            "Robotics": "cs.RO",
            "Neural and Evolutionary Computing": "cs.NE",
            "Information Retrieval": "cs.IR",
            "Human-Computer Interaction": "cs.HC",
            "Distributed, Parallel, and Cluster Computing": "cs.DC",
            "Software Engineering": "cs.SE",
            "Cryptography and Security": "cs.CR",
            "Data Structures and Algorithms": "cs.DS",
        }
        query_parts = [f"cat:{category_codes[category]}"]
        if keywords := keywords.strip().replace('"', ""):
            query_parts.append(f'all:"{keywords}"')
        if author := author.strip().replace('"', ""):
            query_parts.append(f'au:"{author}"')

        now = datetime.now(timezone.utc)
        lookback_hours = 72 if now.weekday() == 0 else 24
        since = now - timedelta(hours=lookback_hours)
        query_parts.append(
            f"submittedDate:[{since:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"
        )

        query = " AND ".join(query_parts)
        papers = fetch_papers(query, paper_count)
        lines = [
            "# Daily arXiv Digest",
            "",
            f"Category: **{category}** · Last {lookback_hours} hours",
            "",
        ]

        if not papers:
            lines.append("No matching papers found.")

        for paper in papers:
            lines.extend(
                [
                    f"## [{paper['title']}]({paper['url']})",
                    "",
                    f"**{paper['authors']}** · {paper['published']}",
                    "",
                    _shorten(paper["abstract"], abstract_words),
                    "",
                ]
            )

        print(f"Found {len(papers)} papers for {query!r}")
        return Markdown(content="\n".join(lines))
