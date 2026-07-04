"""Publish a weekday digest of recent arXiv papers."""

from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from mixtrain import Markdown, MixRoutine, on_schedule


ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}


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
        query: str = "cat:cs.LG",
        paper_count: int = 5,
        abstract_words: int = 60,
    ) -> Markdown:
        if not 1 <= paper_count <= 20:
            raise ValueError("paper_count must be between 1 and 20")
        if not 20 <= abstract_words <= 200:
            raise ValueError("abstract_words must be between 20 and 200")

        papers = fetch_papers(query, paper_count)
        lines = ["# Daily arXiv Digest", "", f"Query: `{query}`", ""]

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
