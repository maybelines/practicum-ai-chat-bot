import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "Naruto Uzumaki", "Sasuke Uchiha", "Sakura Haruno", "Kakashi Hatake",
    "Hinata Hyūga", "Shikamaru Nara", "Gaara", "Rock Lee",
    "Neji Hyūga", "Tsunade", "Jiraiya", "Orochimaru",
    "Itachi Uchiha", "Minato Namikaze", "Kushina Uzumaki", "Obito Uchiha",
    "Konohagakure", "Sunagakure", "Kirigakure", "Kumogakure",
    "Iwagakure", "Amegakure", "Chakra", "Rasengan",
    "Chidori", "Sharingan", "Byakugan", "Shadow Clone Technique",
    "Akatsuki", "Anbu", "Kunai", "Shuriken",
]


def get_article(title, section=0):
    suffix = f"_{section}" if section else ""
    cache = ROOT / ".cache/naruto" / (title.replace(" ", "_") + suffix + ".html")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    params = urlencode({"action": "parse", "page": title, "prop": "text",
                        "section": section, "format": "json", "redirects": 1})
    request = Request("https://naruto.fandom.com/api.php?" + params,
                      headers={"User-Agent": "KnowledgeBaseStudy/1.0"})
    with urlopen(request, timeout=45) as response:
        data = json.load(response)
    if "error" in data:
        raise ValueError(f"{title}: {data['error']}")
    html = data["parse"]["text"]["*"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(html, encoding="utf-8")
    time.sleep(0.5)
    return html


def clean_article(html):
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(".mw-parser-output")
    if body is None:
        raise ValueError("Article body not found")
    for extra in body.select("table, aside, sup, script, style, figure, .thumb, .gallery, h2, h3, .mw-references-wrap"):
        extra.decompose()
    text = body.get_text(" ", strip=True)
    # В скобках обычно идут японские написания и переводы названий.
    while re.search(r"\([^()]*\)", text):
        text = re.sub(r"\([^()]*\)", "", text)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?'’])", r"\1", text)
    text = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.replace(", hence the name", "").replace("As its name suggests, it", "It")
    text = text.replace("Befitting its name, the village", "The village")
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    text = " ".join(sentence for sentence in sentences
                    if not re.search(r"\b(?:anime|manga|episode|novel|video game|kanji|series)\b", sentence, re.I))
    if not text:
        raise ValueError("No article text found")
    return text


def main():
    terms = json.loads((ROOT / "terms_map.json").read_text(encoding="utf-8"))
    replacements = {key.casefold(): value for key, value in terms.items()}
    # Сначала заменяем полные имена, затем короткие.
    names = sorted(terms, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(name) for name in names) + r")(?!\w)", re.I)
    output = ROOT / "knowledge_base"
    output.mkdir(exist_ok=True)
    for title in PAGES:
        text = clean_article(get_article(title))
        if len(text.split()) < 40:
            detail = clean_article(get_article(title, section=1))
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", detail)
            text += "\n\n" + " ".join(sentences[:4])
        text = pattern.sub(lambda match: replacements[match.group().casefold()], text)
        text = re.sub(r"\ban (?=[BCDFGHJKLMNPQRSTVWXYZ])", "a ", text)
        text = re.sub(r"\ba (inherited|ancient|initiate)\b", r"an \1", text)
        text = re.sub(r"\bthe the\b", "the", text)
        name = terms[title]
        filename = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".md"
        # Код нужен для проверки: его нет в исходной статье.
        code = hashlib.sha256(("elvar-archive:" + name).encode()).hexdigest()[:12].upper()
        document = f"# {name}\n\n{text}\n\nArchive code: {code}.\n"
        (output / filename).write_text(document, encoding="utf-8")
        print(f"Saved {filename}", flush=True)
    print(f"Done: {len(PAGES)} documents")


if __name__ == "__main__":
    main()
