"""核验 refs.json 中每条参考文献是否真实存在、标识符是否挂对。

用法: python3 verify.py <cond> [<cond> ...]
每条文献的判定：
  id_ok       DOI/arXiv 号能解析，且解析到的标题与所填标题一致
  id_wrong    标识符能解析，但指向另一篇论文（挂错号）
  id_broken   填了标识符但解析不到
  title_only  没有可用标识符，但按标题能在 OpenAlex/Crossref 找到同名论文
  not_found   标识符和标题都查不到（疑似编造，需人工复核）
"""
import json, re, sys, time, difflib, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent
UA = {"User-Agent": "citation-check/0.1 (mailto:research-check@example.org)"}
CACHE_FILE = ROOT / "verify_cache.json"
CACHE = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


FAILED = set()
OPENALEX_DOWN = [False]
_last_arxiv = [0.0]


def get(url, as_json=True):
    """取 URL；404 视为不存在（返回 None 并缓存），网络/限流失败记入 FAILED，不当作不存在。"""
    if url in CACHE:
        return CACHE[url]
    if "api.openalex.org" in url and OPENALEX_DOWN[0]:
        FAILED.add(url)
        return None
    for attempt in range(5):
        if "export.arxiv.org" in url:  # arXiv 要求请求间隔 ≥3 秒
            wait = 3.1 - (time.time() - _last_arxiv[0])
            if wait > 0:
                time.sleep(wait)
            _last_arxiv[0] = time.time()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode("utf-8", "replace")
            val = json.loads(body) if as_json else body
            if not as_json and "export.arxiv.org" in url and "<entry>" not in body and "<feed" not in body:
                raise ValueError("arXiv 返回异常内容")
            CACHE[url] = val
            FAILED.discard(url)
            return val
        except urllib.error.HTTPError as e:
            if e.code == 404:
                CACHE[url] = None
                return None
            if e.code == 429 and "api.openalex.org" in url:  # 当日配额用尽，改走 Crossref/arXiv
                OPENALEX_DOWN[0] = True
                FAILED.add(url)
                return None
            time.sleep(5 * 2 ** attempt)
        except Exception:
            time.sleep(5 * 2 ** attempt)
    FAILED.add(url)
    return None


def norm(t):
    t = (t or "").lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^a-z0-9一-鿿]+", " ", t)
    return " ".join(t.split())


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return max(0.9, difflib.SequenceMatcher(None, a, b).ratio())
    return difflib.SequenceMatcher(None, a, b).ratio()


def clean_doi(d):
    if not d:
        return None
    d = str(d).strip()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d, flags=re.I)
    return d if d.startswith("10.") else None


def clean_arxiv(a, url=None):
    for s in (a, url if url and "arxiv.org" in str(url) else None):
        if not s:
            continue
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", str(s)) or re.search(r"([a-z\-]+(\.[A-Z]{2})?/\d{7})", str(s))
        if m:
            return m.group(1)
    return None


def openalex_doi(doi):
    w = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
    if w:
        return {"title": w.get("title"), "year": w.get("publication_year"),
                "authors": [a["author"]["display_name"] for a in w.get("authorships", [])]}
    c = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if c and c.get("message"):
        m = c["message"]
        yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        return {"title": (m.get("title") or [""])[0], "year": yr,
                "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in m.get("author", [])]}
    return None


def arxiv_lookup(aid):
    x = get("http://export.arxiv.org/api/query?id_list=" + aid, as_json=False)
    if not x:
        return None
    try:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = ET.fromstring(x).find("a:entry", ns)
        if e is None or e.find("a:title", ns) is None:
            return None
        title = " ".join(e.find("a:title", ns).text.split())
        if title.lower() == "error":
            return None
        return {"title": title, "year": int(e.find("a:published", ns).text[:4]),
                "authors": [n.find("a:name", ns).text for n in e.findall("a:author", ns)]}
    except Exception:
        return None


def title_search(title):
    best = None
    r = get("https://api.openalex.org/works?per_page=5&search=" + urllib.parse.quote(title))
    for w in (r or {}).get("results", []):
        s = sim(title, w.get("title"))
        if not best or s > best[0]:
            best = (s, {"title": w.get("title"), "year": w.get("publication_year"),
                        "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
                        "doi": w.get("doi")})
    if not best or best[0] < 0.9:
        r = get("https://api.crossref.org/works?rows=5&query.bibliographic=" + urllib.parse.quote(title))
        for m in ((r or {}).get("message") or {}).get("items", []):
            t = (m.get("title") or [""])[0]
            s = sim(title, t)
            if not best or s > best[0]:
                yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
                best = (s, {"title": t, "year": yr, "doi": m.get("DOI"),
                            "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in m.get("author", [])]})
    if not best or best[0] < 0.9:
        q = urllib.parse.quote(f'ti:"{title}"')
        x = get(f"http://export.arxiv.org/api/query?max_results=3&search_query={q}", as_json=False)
        if x:
            try:
                ns = {"a": "http://www.w3.org/2005/Atom"}
                for e in ET.fromstring(x).findall("a:entry", ns):
                    t = " ".join(e.find("a:title", ns).text.split())
                    s = sim(title, t)
                    if not best or s > best[0]:
                        best = (s, {"title": t, "year": int(e.find("a:published", ns).text[:4]),
                                    "authors": [n.find("a:name", ns).text for n in e.findall("a:author", ns)]})
            except Exception:
                pass
    return best


def surname(name):
    name = (name or "").strip()
    if "," in name:
        return norm(name.split(",")[0])
    parts = norm(name).split()
    return parts[-1] if parts else ""


def meta_issues(ref, rec):
    issues = []
    try:
        if rec.get("year") and ref.get("year") and abs(int(ref["year"]) - int(rec["year"])) > 1:
            issues.append(f"年份 {ref['year']}≠{rec['year']}")
    except (TypeError, ValueError):
        pass
    ra = ref.get("authors") or []
    if isinstance(ra, str):
        ra = re.split(r",|;| and ", ra)
    if ra and rec.get("authors"):
        mine = surname(ra[0])
        theirs = {surname(a) for a in rec["authors"]} | {norm(a) for a in rec["authors"]}
        if mine and not any(mine in t or t in mine for t in theirs if t):
            issues.append(f"第一作者 {ra[0]} 不在作者列表")
    return issues


def check(ref):
    title = ref.get("title") or ""
    out = {"id": ref.get("id"), "title": title}
    url = ref.get("url") or ""
    doi = clean_doi(ref.get("doi"))
    if not doi:
        m = re.search(r"(?:doi\.org/|/doi/(?:abs/|full/|pdf/)?)(10\.\d{4,9}/[^\s?#]+)", url)
        doi = clean_doi(m.group(1)) if m and not m.group(1).startswith("10.5555") else None
    aid = clean_arxiv(ref.get("arxiv"), url)
    gh = re.match(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", url)
    if gh and not doi and not aid:
        r = get(f"https://api.github.com/repos/{gh.group(1)}/{gh.group(2).removesuffix('.git')}")
        out.update(status="software_ok" if r and r.get("full_name") else "not_found", via=url)
        return out
    orv = re.search(r"openreview\.net/(?:forum|pdf)\?id=([\w-]+)", url)
    if orv and not doi and not aid:
        for api in ("https://api2.openreview.net/notes?id=", "https://api.openreview.net/notes?id="):
            r = get(api + orv.group(1))
            notes = (r or {}).get("notes") or []
            if notes:
                t = notes[0].get("content", {}).get("title")
                t = t.get("value") if isinstance(t, dict) else t
                sc = sim(title, t)
                out.update(status="id_ok" if sc >= 0.85 else "id_wrong", via=url, sim=round(sc, 2), points_to=t, issues=[])
                return out
    tried, unchecked = [], []
    for kind, key, fn in (("doi", doi, openalex_doi), ("arxiv", aid, arxiv_lookup)):
        if not key:
            continue
        before = set(FAILED)
        rec = fn(key)
        if rec is None:
            if FAILED - before:
                unchecked.append(f"{kind}:{key}")
            else:
                tried.append((kind, key, None, 0))
            continue
        s = sim(title, rec["title"])
        tried.append((kind, key, rec, s))
        if s >= 0.85:
            out.update(status="id_ok", via=f"{kind}:{key}", sim=round(s, 2), issues=meta_issues(ref, rec))
            return out
    wrong = [t for t in tried if t[2] is not None]
    best = title_search(title) if title else None
    if wrong:
        k, key, rec, s = wrong[0]
        out.update(status="id_wrong", via=f"{k}:{key}", sim=round(s, 2), points_to=rec["title"],
                   title_exists=bool(best and best[0] >= 0.9))
        return out
    if best and best[0] >= 0.9:
        out.update(status="title_only" if not tried else "id_broken", sim=round(best[0], 2), unchecked=unchecked,
                   matched=best[1]["title"], issues=meta_issues(ref, best[1]),
                   title_exists=True, broken=[f"{k}:{v}" for k, v, *_ in tried])
        return out
    if unchecked and not tried:
        out.update(status="unchecked", unchecked=unchecked, best_guess=(best[1]["title"] if best else None))
        return out
    out.update(status="id_broken" if tried else "not_found",
               best_guess=(best[1]["title"] if best else None), best_sim=round(best[0], 2) if best else 0,
               title_exists=False, broken=[f"{k}:{v}" for k, v, *_ in tried])
    return out


def main():
    for cond in sys.argv[1:]:
        p = ROOT / cond / "refs.json"
        if not p.exists():
            print(cond, "无 refs.json"); continue
        refs = json.loads(p.read_text())
        res = []
        for r in refs:
            res.append(check(r))
            CACHE_FILE.write_text(json.dumps(CACHE))
        (ROOT / f"{cond}.verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        cnt = {}
        for r in res:
            cnt[r["status"]] = cnt.get(r["status"], 0) + 1
        real = sum(1 for r in res if r["status"] in ("id_ok", "title_only", "software_ok") or r.get("title_exists"))
        meta = sum(1 for r in res if r.get("issues"))
        print(f"== {cond}: {len(res)} 条 | {cnt} | 论文真实存在 {real}/{len(res)} | 元数据有误 {meta}")
        for r in res:
            if r["status"] != "id_ok" or r.get("issues"):
                print("  ", r["id"], r["status"], "|", r["title"][:70], "|",
                      r.get("points_to") or r.get("matched") or r.get("best_guess") or "", "|", r.get("issues") or "")


if __name__ == "__main__":
    main()
