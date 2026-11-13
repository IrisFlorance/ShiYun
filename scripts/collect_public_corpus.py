"""Assemble the first versioned Chinese corpus from the pinned public sources."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from urllib.parse import quote, unquote
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from bs4 import BeautifulSoup

from ald.data.corpus import normalize
from ald.utils.files import atomic_write, digest, write_json, write_jsonl

CLASSICAL_REPO = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry"
CLASSICAL_COMMIT = "b8594f81a89752241442f2ce267d6f66f96704ee"
WIKI = "https://zh.wikisource.org"
AGENT = "ALD-corpus/0.1 (local research; source https://github.com/chinese-poetry/chinese-poetry)"
CLASSICAL = {
    "tang": (
        900,
        [
            f"全唐诗/poet.tang.{i}.json"
            for i in (0, 5000, 11000, 17000, 23000, 29000, 37000, 44000, 51000, 56000)
        ],
    ),
    "song_ci": (400, [f"宋词/ci.song.{i}.json" for i in (0, 5000, 10000, 15000, 20000)]),
    "yuan_qu": (140, ["元曲/yuanqu.json"]),
    "shijing": (100, ["诗经/shijing.json"]),
}
AUTHORS = {
    "徐志摩": "徐志摩",
    "聞一多": "闻一多",
    "戴望舒": "戴望舒",
    "朱自清": "朱自清",
    "劉半農": "刘半农",
    "胡適": "胡适",
    "魯迅": "鲁迅",
    "劉大白": "刘大白",
    "楊華": "杨华",
    "朱湘": "朱湘",
    "沈尹默": "沈尹默",
    "康白情": "康白情",
}
PROSE_BOOKS = {"背影_(散文集)": "朱自清", "朝花夕拾": "鲁迅"}
NARRATIVE_BOOKS = {"吶喊": "鲁迅", "彷徨": "鲁迅"}
OPEN_POEMS = (
    "死水",
    "再別康橋",
    "偶然",
    "雨巷",
    "小河",
    "翡冷翠的一夜",
    "七子之歌",
    "萧萧雨",
    "纸船",
    "教我如何不想她",
)
POEM_SERIES = ("小詩十二首", "山花_(楊華)", "晨光集", "黑潮集", "七子之歌")


def fetch(url, path):
    if path.is_file() and path.stat().st_size:
        return path.read_bytes()
    request = Request(url, headers={"User-Agent": AGENT})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=8) as response:
            return response.read()
    except (TimeoutError, ConnectionError):
        with urlopen(request, timeout=14) as response:
            return response.read()


def read_wiki(folder, title):
    filename = sha256(title.encode()).hexdigest()[:24] + ".html"
    cached = folder / "wiki" / filename
    local = folder / (title.replace(":", "_") + ".html")
    if re.search(r'[<>"/\\|?*]', title) or not local.exists():
        local = cached
    url = f"{WIKI}/zh-hans/{quote(title, safe='')}"
    try:
        raw = fetch(url, local)
    except OSError as error:
        return title, None, None, str(error)
    soup = BeautifulSoup(raw, "html.parser")
    content = max(
        soup.select(".mw-parser-output"), key=lambda tag: len(tag.get_text()), default=None
    )
    rev = re.search(rb'"wgRevisionId":(\d+)', raw)
    if content is None or not rev or rev[1] == b"0":
        return title, None, None, "no work content or revision"
    if not local.exists():
        return title, content, (int(rev[1]), raw, cached), None
    return title, content, (int(rev[1]), raw, local), None


def links(root):
    for anchor in root.select("a[href^='/wiki/']"):
        raw = unquote(anchor["href"].removeprefix("/wiki/"))
        if ":" not in raw and "#" not in raw and raw.strip():
            yield raw


def author_links(folder, source):
    _, content, _, _ = read_wiki(folder, f"Author:{source}")
    if content is None:
        return []
    discovered = []
    enabled = False
    for element in content.select("h2,h3,ul"):
        if element.name.startswith("h"):
            heading = element.get_text(strip=True)
            if element.name == "h2" or heading in {"作品", "译作", "譯作"}:
                enabled = heading in {"新诗", "新詩", "诗", "詩", "作品", "新诗作品", "新詩作品"}
        elif enabled:
            discovered.extend(links(element))
    return discovered


def verified_pd(content, *, inherited=False):
    box = content.select_one(".licenseContainer")
    if box is None:
        return inherited
    paragraphs = [node.get_text(" ", strip=True) for node in box.find_all("p")]
    notice = next(
        (value for value in paragraphs if "公有领域" in value), box.get_text(" ", strip=True)
    )
    if any(term in notice for term in ("然而", "但美国", "但 美国", "非公有", "不属于", "仍然")):
        return False
    if inherited:
        return True
    return (
        "公有领域" in notice
        and "美国" in notice
        and ("1931年1月1日以前" in notice or "1996年1月1日" in notice)
    )


def body_text(element):
    for node in element.select(
        ".reference,.pagenum,.wst-annotation,.wst-tooltip,.ws-noexport,.mw-editsection,"
        ".mw-ref,.noprint,.dynlayout-exempt,rt,rp,.inline-note"
    ):
        node.decompose()
    for node in element.find_all("br"):
        if node.next_sibling is not None and str(node.next_sibling).startswith("\n"):
            node.decompose()
        else:
            node.replace_with("\n")
    return normalize(element.get_text().replace("\u200b", "").replace("\u200e", ""))


def wiki_author(content):
    header = content.select_one("#headerContainer, .ws-header")
    if header is None:
        return None
    for anchor in header.select("a[href*='Author:']"):
        name = unquote(anchor["href"]).rsplit("Author:", 1)[1].replace("_", " ")
        if name in AUTHORS:
            return AUTHORS[name]
    return None


def paragraphs(content):
    root = content.select_one(".prp-pages-output") or content
    return [body_text(node) for node in root.find_all("p", recursive=False)]


def series_poems(title, content):
    if title in {"小詩十二首", "山花_(楊華)"}:
        markers = (
            r"(?m)^\d{1,2}、?[ \t]*$" if title == "小詩十二首" else r"(?m)^山花之\d{1,3}[ \t]*$"
        )
        return [normalize(value) for value in re.split(markers, "\n".join(paragraphs(content)))[1:]]
    if title in {"晨光集", "黑潮集"}:
        result = []
        ending = r"(?:--|——)\s*(?:晨光集|黑潮集)\s*\d+\s*$"
        for node in content.select("li"):
            text = body_text(node)
            if re.search(ending, text):
                result.append(re.sub(ending, "", text).strip())
        return result
    if title == "七子之歌":
        return [body_text(node) for node in content.select(".poem")[:7]]
    raise ValueError(f"Unknown series: {title}")


def chunk_sentences(text, *, limit=160):
    pieces = re.findall(r"[^。！？；!?\n]+[。！？；!?]*", text)
    buffer = ""
    for piece in pieces:
        piece = piece.strip()
        if len(buffer) + len(piece) > limit and buffer:
            if len(buffer) >= 22:
                yield buffer
            buffer = ""
        while len(piece) > limit:
            yield piece[:limit]
            piece = piece[limit:]
        buffer += piece
    if len(buffer) >= 22:
        yield buffer


def wiki_row(title, revision, author, form, text, suffix=""):
    return {
        "text": text,
        "title": title,
        "author": author,
        "form": form,
        "work_id": f"wikisource:{title}",
        "source": f"{WIKI}/w/index.php?title={quote(title, safe='')}&oldid={revision}{suffix}",
        "source_revision": str(revision),
        "license": "Public-domain original; Wikisource revision attribution",
    }


def modern_and_prose(downloads):
    category = downloads / "Category_新詩.html"
    candidates = set((*OPEN_POEMS, *POEM_SERIES))
    raw = fetch(f"{WIKI}/zh-hans/{quote('Category:新詩', safe='')}", category)
    if not category.exists():
        atomic_write(category, lambda temp: temp.write_bytes(raw))
    soup = BeautifulSoup(raw, "html.parser")
    for block in soup.select("#mw-pages .mw-category"):
        candidates.update(links(block))
    with ThreadPoolExecutor(4) as pool:
        for pages in pool.map(lambda author: author_links(downloads, author), AUTHORS):
            candidates.update(pages)
    print(f"Wiki candidate pages: {len(candidates)}", flush=True)
    inherited = {}
    for title, author in {"野艸": "鲁迅", **PROSE_BOOKS, **NARRATIVE_BOOKS}.items():
        _, content, _, error = read_wiki(downloads, title)
        if error:
            print(f"collection unavailable: {title}: {error}", flush=True)
            continue
        if title in PROSE_BOOKS:
            root = content.select_one(".ws-summary") or content.select_one("ul")
        else:
            root = content.select_one(".ws-summary")
        if root is not None:
            for page in links(root):
                if page not in {"小引", "後記", "后记", "《背影》序", "《梅花》後記"}:
                    inherited[page] = (author, "prose_poem" if title == "野艸" else "prose")
    candidates.update(inherited)
    result = []
    prose = []
    rejected = Counter()
    sources = []
    with ThreadPoolExecutor(5) as pool:
        for i, entry in enumerate(
            pool.map(lambda title: read_wiki(downloads, title), sorted(candidates)), 1
        ):
            title, content, origin, error = entry
            if i % 40 == 0:
                print(
                    f"wiki {i}/{len(candidates)}: {len(result)} poem rows, {len(prose)} prose passages",
                    flush=True,
                )
            if error:
                rejected["unavailable"] += 1
                continue
            author = wiki_author(content)
            expected, form = inherited.get(title, (None, "new_poem"))
            if author is None:
                rejected["unverified_author"] += 1
                continue
            if expected is not None and expected != author:
                rejected["author_mismatch"] += 1
                continue
            if not verified_pd(content, inherited=expected is not None):
                rejected["license_or_edition_unverified"] += 1
                continue
            revision, raw, path = origin
            if title in POEM_SERIES:
                for ordinal, text in enumerate(series_poems(title, content), 1):
                    if 12 <= len(text) <= 520:
                        record = wiki_row(title, revision, author, form, text)
                        record["work_id"] += f":{ordinal}"
                        record["title"] += f" [{ordinal}]"
                        result.append(record)
            elif form == "new_poem":
                poem_blocks = [
                    node
                    for node in content.select(".poem")
                    if node.find_parent(class_="poem") is None
                ]
                if poem_blocks:
                    values = [body_text(node) for node in poem_blocks]
                else:
                    blocks = content.find_all("p", recursive=False)
                    if not any(node.find("br") for node in blocks):
                        rejected["no_structured_verse"] += 1
                        continue
                    values = [body_text(node) for node in blocks]
                text = normalize("\n\n".join(value for value in values if value and value != title))
                if not 12 <= len(text) <= 520:
                    rejected["verse_length"] += 1
                    continue
                result.append(wiki_row(title, revision, author, form, text))
            else:
                parts = [value for value in paragraphs(content) if len(value) >= 12]
                snippets = list(chunk_sentences("\n".join(parts)))
                if not snippets:
                    rejected["no_prose_text"] += 1
                    continue
                if form == "prose_poem":
                    result.extend(
                        wiki_row(title, revision, author, form, text) for text in snippets[:4]
                    )
                else:
                    prose.extend(
                        wiki_row(title, revision, author, form, text) for text in snippets[:8]
                    )
            if not path.exists():
                atomic_write(path, lambda temp: temp.write_bytes(raw))
            sources.append(
                {
                    "title": title,
                    "revision": revision,
                    "sha256": sha256(raw).hexdigest(),
                    "author": author,
                    "form": form,
                }
            )
    return result, prose, sources, dict(rejected)


def classical(folder, seed):
    root = f"{CLASSICAL_REPO}/{CLASSICAL_COMMIT}"
    license_path = folder / "LICENSE"
    license_bytes = fetch(f"{root}/LICENSE", license_path)
    if b"MIT License" not in license_bytes:
        raise ValueError("Unexpected upstream license")
    if not license_path.exists():
        atomic_write(license_path, lambda temp: temp.write_bytes(license_bytes))
    rng = random.Random(seed)
    selected = []
    manifest = []
    for form, (count, paths) in CLASSICAL.items():
        pool = []
        for filename in paths:
            url = f"{root}/{quote(filename)}"
            destination = folder / Path(filename).name
            raw = fetch(url, destination)
            if not destination.exists():
                atomic_write(destination, lambda temp: temp.write_bytes(raw))
            manifest.append({"source": url, "sha256": sha256(raw).hexdigest()})
            for i, record in enumerate(json.loads(raw)):
                lines = record.get("paragraphs") or record.get("content")
                if isinstance(lines, str):
                    lines = re.split(r"(?<=[。！？])", lines)
                if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
                    continue
                text = normalize("\n".join(lines))
                if not 16 <= len(text) <= 165 or any(ch in text for ch in "□�"):
                    continue
                author = record.get("author") or ""
                if author in {"佚名", "無名氏", "无名氏", "不詳", "不详", "无名", "無名"}:
                    author = ""
                title = record.get("title") or record.get("rhythmic") or ""
                pool.append(
                    {
                        "text": text,
                        "title": title,
                        "author": author,
                        "form": form,
                        "work_id": f"chinese-poetry:{filename}:{i}",
                        "source": url,
                        "source_revision": CLASSICAL_COMMIT,
                        "license": "Public-domain original; upstream MIT",
                    }
                )
        rng.shuffle(pool)
        per_author = Counter()
        group = []
        for item in pool:
            author = item["author"]
            if author and per_author[author] >= 30:
                continue
            group.append(item)
            per_author[author] += 1
            if len(group) == count:
                break
        if len(group) != count:
            raise ValueError(f"Not enough eligible {form} records: {len(group)}/{count}")
        selected.extend(group)
    return selected, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    folder = args.out / "downloads"
    older, files = classical(folder, args.seed)
    modern, prose, pages, rejected = modern_and_prose(folder)
    if len(modern) < 35 or len({row["author"] for row in modern}) < 3 or len(prose) < 20:
        raise RuntimeError(
            f"Insufficient verified modern material: poems={len(modern)}, prose={len(prose)}, {rejected}"
        )
    write_jsonl(args.out / "poems.jsonl", modern + older)
    write_jsonl(args.out / "prose.jsonl", prose)
    manifest = {
        "seed": args.seed,
        "classical_repository": CLASSICAL_REPO,
        "classical_commit": CLASSICAL_COMMIT,
        "classical_license_sha256": digest(folder / "LICENSE"),
        "classical_files": files,
        "wikimedia_pages": pages,
        "wiki_rejected": rejected,
        "counts": dict(Counter(row["form"] for row in modern + older + prose)),
        "poems_sha256": digest(args.out / "poems.jsonl"),
        "prose_sha256": digest(args.out / "prose.jsonl"),
    }
    write_json(args.out / "sources.json", manifest)
    print(manifest["counts"], flush=True)


if __name__ == "__main__":
    main()
