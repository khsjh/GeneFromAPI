# =========================
# backend/main.py
# =========================
# Run: uvicorn main:app --reload --host 127.0.0.1 --port 8000

from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import urllib.parse
import asyncio
import os
import re
from pydantic import BaseModel

# ---- Tunables (env overrides allowed) ----
DEFAULT_TIMEOUT = float(os.getenv("GLE_DEFAULT_TIMEOUT", "20.0"))    # 기본 httpx 타임아웃
HZ_TIMEOUT      = float(os.getenv("GLE_HZ_TIMEOUT", "80.0"))         # Harmonizome 전용 타임아웃
OT_TIMEOUT      = float(os.getenv("GLE_OT_TIMEOUT", "25.0"))         # OpenTargets 전용 타임아웃
RETRIES         = int(os.getenv("GLE_RETRIES", "3"))                 # 재시도 횟수

ENSEMBL_LOOKUP   = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/"
HARMONIZOME_BASE = "https://maayanlab.cloud/Harmonizome/api/1.0/gene/"
OPENTARGETS_GQL  = "https://api.platform.opentargets.org/api/v4/graphql"
USER_AGENT = "GeneListExplorerFastAPI/1.4"

# ---- Request model (질환 필터 추가) ----
class AggregateRequest(BaseModel):
    symbols: List[str]
    topn: Optional[int] = 15
    fetch_harmonizome: bool = True
    fetch_opentargets: bool = True
    disease_query: Optional[str] = None     # 질환 검색어
    match_mode: Optional[str] = "auto"      # "auto" | "simple" | "exact"

# ---------------- HTTP helper ----------------
class HttpJson:
    def __init__(self):
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=20)
        self.client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            limits=limits,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            http2=True,
        )

    async def get(self, url: str, timeout: Optional[float] = None) -> Any:
        for attempt in range(RETRIES):
            try:
                r = await self.client.get(url, timeout=timeout)
                r.raise_for_status()
                try:
                    return r.json()
                except Exception:
                    text = (r.text or "")[:200]
                    raise RuntimeError(f"Non-JSON response from GET {url}: {text}")
            except Exception:
                if attempt == RETRIES - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

    async def post_json(self, url: str, body: Dict[str, Any], timeout: Optional[float] = None) -> Any:
        for attempt in range(RETRIES):
            try:
                r = await self.client.post(
                    url,
                    json=body,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    timeout=timeout,
                )
                r.raise_for_status()
                try:
                    return r.json()
                except Exception:
                    text = (r.text or "")[:200]
                    raise RuntimeError(f"Non-JSON response from POST {url}: {text}")
            except Exception:
                if attempt == RETRIES - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

http = HttpJson()

app = FastAPI(title="Gene List Explorer API", default_response_class=ORJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------------- utils ----------------
def clean_symbols(symbols: List[str]) -> List[str]:
    seen, out = set(), []
    for s in symbols or []:
        s2 = (s or "").strip()
        if s2 and s2 not in seen:
            seen.add(s2)
            out.append(s2)
    return out

def _normalize_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'")

def _is_exact_mode(query: Optional[str], mode: str) -> bool:
    if not query:
        return False
    if mode == "exact":
        return True
    if mode == "simple":
        return False
    # auto: 따옴표로 감싸져 있으면 exact
    q = query.strip()
    return (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'"))

def _match_text(name: str, query: str, mode: str) -> bool:
    """case-insensitive; simple=부분/단어 매치, exact=정확히 동일"""
    if not query:
        return True
    name_l = (name or "").lower()
    q_l = (_normalize_quotes(query) if _is_exact_mode(query, mode) else query).lower()
    if _is_exact_mode(query, mode):
        return name_l == q_l
    # simple: 단순 부분매치 + 단어경계 우선
    # 단어경계 시도
    try:
        if re.search(rf"\b{re.escape(q_l)}\b", name_l):
            return True
    except re.error:
        pass
    return q_l in name_l

# ---------------- external fetchers ----------------
async def map_symbol_to_ensembl(sym: str) -> Dict[str, Optional[str]]:
    url = f"{ENSEMBL_LOOKUP}{urllib.parse.quote(sym, safe='')}?content-type=application/json"
    try:
        j = await http.get(url)
        return {"symbol": sym, "ensembl_id": j.get("id"), "matched_symbol": j.get("display_name")}
    except Exception as e:
        print("Ensembl error:", repr(e))
        return {"symbol": sym, "ensembl_id": None, "matched_symbol": None}

# Harmonizome: CTD만 사용 + disease_query 매칭 (순수 질환명으로 매칭)
async def fetch_harmonizome(symbol: str, disease_query: Optional[str], match_mode: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Returns (rows, error_message)
    For CTD Gene-Disease Associations only.
    rows: [{symbol, term, href, threshold, standardized}, ...]
    """
    url = f"{HARMONIZOME_BASE}{urllib.parse.quote(symbol, safe='')}?showAssociations=true"
    CTD_TOKEN = "CTD+Gene-Disease+Associations"
    try:
        j = await http.get(url, timeout=HZ_TIMEOUT)
        assocs = j.get("associations") or []
        out: List[Dict[str, Any]] = []
        for a in assocs:
            gs = (a or {}).get("geneSet") or {}
            href = gs.get("href") or ""
            name = gs.get("name") or ""
            # 1) CTD만 통과 (href에 토큰 포함)
            if CTD_TOKEN not in href:
                continue
            # 2) CTD name → 순수 질환명으로 정규화
            #    예: "Fibrosis/CTD Gene-Disease Associations" → "Fibrosis"
            pure_term = name.split("/CTD Gene-Disease Associations")[0].strip() or name.strip()
            # 3) 질환 필터 적용 (있을 때만)
            if disease_query and not _match_text(pure_term, disease_query, match_mode):
                continue
            # 4) 절대 URL 보정
            if href.startswith("/"):
                href = "https://maayanlab.cloud" + href
            out.append({
                "symbol": j.get("symbol", symbol),
                "term": pure_term,  # 화면 표시는 순수 질환명
                "href": href,
                "threshold": a.get("thresholdValue"),
                "standardized": a.get("standardizedValue"),
            })
        # 점수 높은 순 정렬
        out.sort(key=lambda x: (x.get("standardized") is not None, x.get("standardized") or -1), reverse=True)
        return out, None
    except Exception as e:
        msg = f"Harmonizome error for {symbol}: {repr(e)}"
        print(msg)
        return [], msg

OT_QUERY = (
    "query targetAssoc($ensemblId: String!, $size: Int!, $index: Int!) {"
    "  target(ensemblId: $ensemblId) {"
    "    id approvedSymbol"
    "    associatedDiseases(page: {size: $size, index: $index}) {"
    "      count"
    "      rows { score disease { id name } datatypeScores { id score } datasourceScores { id score } }"
    "    }"
    "  }"
    "}"
)

async def fetch_opentargets_assocs(ensembl_id: str, disease_query: Optional[str], match_mode: str, page_size: int = 200) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not ensembl_id:
        return [], None
    idx, out = 0, []
    try:
        while True:
            body = {"query": OT_QUERY, "variables": {"ensemblId": ensembl_id, "size": page_size, "index": idx}}
            j = await http.post_json(OPENTARGETS_GQL, body, timeout=OT_TIMEOUT)
            data = (j.get("data") or {}).get("target") or {}
            assoc = (data.get("associatedDiseases") or {})
            rows = assoc.get("rows") or []
            total = assoc.get("count") or 0
            sym = data.get("approvedSymbol")
            for r in rows:
                disease = (r.get("disease") or {})
                dname = disease.get("name") or ""
                # 질환 필터 (있으면 적용)
                if disease_query and not _match_text(dname, disease_query, match_mode):
                    continue
                out.append({
                    "ensembl_id": ensembl_id,
                    "approvedSymbol": sym,
                    "disease_id": disease.get("id"),
                    "disease_name": dname,
                    "score": r.get("score"),
                    "datatypeScores": r.get("datatypeScores") or [],
                    "datasourceScores": r.get("datasourceScores") or [],
                })
            idx += 1
            if idx * page_size >= total or not rows:
                break
        return out, None
    except Exception as e:
        msg = f"OpenTargets error for {ensembl_id}: {repr(e)}"
        print(msg)
        return [], msg

# ---------------- routes ----------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """<html><body>
    <h2>Gene List Explorer API</h2>
    <p>Use <code>POST /aggregate</code> with JSON payload.<br>
    See <a href="/docs">/docs</a> or <a href="/redoc">/redoc</a>.</p>
    </body></html>"""

@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "ok"

@app.get("/debug/hz", response_class=JSONResponse)
async def debug_hz(symbol: str, disease_query: Optional[str] = None, match_mode: str = "auto"):
    rows, err = await fetch_harmonizome(symbol, disease_query, match_mode)
    return {"rows": rows, "error": err}

@app.get("/debug/ot", response_class=JSONResponse)
async def debug_ot(ensembl_id: str, disease_query: Optional[str] = None, match_mode: str = "auto"):
    rows, err = await fetch_opentargets_assocs(ensembl_id, disease_query, match_mode)
    return {"rows": rows, "error": err}

@app.post("/aggregate", response_class=JSONResponse)
async def aggregate(req: AggregateRequest):
    symbols = clean_symbols(req.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="No valid symbols provided")

    # 1) map symbols -> Ensembl IDs in parallel
    mappings = await asyncio.gather(*[map_symbol_to_ensembl(s) for s in symbols])

    # 2) per gene tasks (CTD only + disease filter)
    tasks = []
    for m in mappings:
        sym = m["symbol"]
        if req.fetch_harmonizome:
            tasks.append(fetch_harmonizome(sym, req.disease_query, req.match_mode or "auto"))
        if req.fetch_opentargets:
            tasks.append(fetch_opentargets_assocs(m.get("ensembl_id"), req.disease_query, req.match_mode or "auto"))

    results = await asyncio.gather(*tasks)

    # Demultiplex results
    hz: Dict[str, List[Dict[str, Any]]] = {}
    ot: Dict[str, List[Dict[str, Any]]] = {}
    hz_err: Dict[str, Optional[str]] = {}
    ot_err: Dict[str, Optional[str]] = {}

    r_idx = 0
    for m in mappings:
        sym = m["symbol"]
        if req.fetch_harmonizome:
            rows, err = results[r_idx]; r_idx += 1
            hz[sym] = rows; hz_err[sym] = err
        if req.fetch_opentargets:
            rows, err = results[r_idx]; r_idx += 1
            ot[sym] = rows; ot_err[sym] = err

    # Top-N per gene (OpenTargets)
    topn = max(1, req.topn or 15)
    ot_top: Dict[str, List[Dict[str, Any]]] = {}
    for sym, rows in ot.items():
        rows_sorted = sorted(rows, key=lambda x: (x.get("score") is not None, x.get("score") or -1), reverse=True)
        ot_top[sym] = rows_sorted[:topn]

    return {
        "mappings": mappings,
        "harmonizome": hz,
        "harmonizome_errors": hz_err,
        "opentargets": ot,
        "opentargets_errors": ot_err,
        "opentargets_top": ot_top,
    }

