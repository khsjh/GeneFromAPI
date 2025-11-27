# =========================
# backend/main.py (with DrugProtAI)
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
import json
import time
import hashlib
from pathlib import Path
from pydantic import BaseModel
from playwright.async_api import async_playwright

# ---- Tunables (env overrides allowed) ----
DEFAULT_TIMEOUT = float(os.getenv("GLE_DEFAULT_TIMEOUT", "80.0"))
HZ_TIMEOUT      = float(os.getenv("GLE_HZ_TIMEOUT", "80.0"))
OT_TIMEOUT      = float(os.getenv("GLE_OT_TIMEOUT", "80.0"))
RETRIES         = int(os.getenv("GLE_RETRIES", "3"))

# DrugProtAI settings
DRUGPROTAI_HOME = "https://drugprotai.pythonanywhere.com/"
SCRAPER_MIN_INTERVAL = 0.8  # rate limit (초)
SCRAPER_CACHE_TTL = 24 * 3600  # 1일
SCRAPER_CACHE_DIR = Path(os.getenv("SCRAPER_CACHE_DIR", "/tmp/drugprotai_cache"))
DRUGPROTAI_TIMEOUT = 240.0

# 캐시 디렉토리 생성은 앱 시작 시점으로 지연

ENSEMBL_LOOKUP   = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/"
UNIPROT_BASE     = "https://rest.uniprot.org"
HARMONIZOME_BASE = "https://maayanlab.cloud/Harmonizome/api/1.0/gene/"
OPENTARGETS_GQL  = "https://api.platform.opentargets.org/api/v4/graphql"
USER_AGENT = "GeneListExplorerFastAPI/1.5"

# Global state
_last_scrape_ts = 0.0
_drugprotai_sem = asyncio.Semaphore(1)  # 동시 실행 제한

# ---- Request model ----
class AggregateRequest(BaseModel):
    symbols: List[str]
    topn: Optional[int] = 15
    fetch_harmonizome: bool = True
    fetch_opentargets: bool = True
    fetch_drugprotai: bool = True
    disease_query: Optional[str] = None
    match_mode: Optional[str] = "auto"  # "auto" | "simple" | "exact"

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

                # 1차: 정상 JSON이라고 가정
                try:
                    return r.json()
                except Exception:
                    # Harmonizome처럼 "JSON + 기타 텍스트"가 섞인 경우 대비
                    text = r.text or ""

                    # 본문에서 가장 바깥쪽 { ... } 구간만 추출해서 다시 파싱
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        candidate = text[start:end + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            pass  # 아래에서 에러로 떨어짐

                    short = text[:200]
                    raise RuntimeError(
                        f"Non-JSON response from GET {url}: {short}"
                    )

            except Exception:
                if attempt == RETRIES - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

#    async def get(self, url: str, timeout: Optional[float] = None) -> Any:
#        for attempt in range(RETRIES):
#            try:
#                r = await self.client.get(url, timeout=timeout)
#                r.raise_for_status()
#                try:
#                    return r.json()
#                except Exception:
#                    text = (r.text or "")[:200]
#                    raise RuntimeError(f"Non-JSON response from GET {url}: {text}")
#            except Exception:
#                if attempt == RETRIES - 1:
#                    raise
#                await asyncio.sleep(0.5 * (attempt + 1))

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

# 캐시 디렉토리 생성 (앱 초기화 시점)
try:
    SCRAPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory created: {SCRAPER_CACHE_DIR}")
except Exception as e:
    print(f"Warning: Could not create cache directory {SCRAPER_CACHE_DIR}: {e}")

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
    try:
        if re.search(rf"\b{re.escape(q_l)}\b", name_l):
            return True
    except re.error:
        pass
    return q_l in name_l

def _cache_key(key: str) -> Path:
    """캐시 파일 경로 생성"""
    return SCRAPER_CACHE_DIR / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")

# ---------------- external fetchers ----------------
async def map_symbol_to_ensembl(sym: str) -> Dict[str, Optional[str]]:
    url = f"{ENSEMBL_LOOKUP}{urllib.parse.quote(sym, safe='')}?content-type=application/json"
    try:
        j = await http.get(url)
        return {"symbol": sym, "ensembl_id": j.get("id"), "matched_symbol": j.get("display_name")}
    except Exception as e:
        print("Ensembl error:", repr(e))
        return {"symbol": sym, "ensembl_id": None, "matched_symbol": None}

async def fetch_uniprot_id(sym: str) -> Dict[str, Optional[str]]:
    """Gene symbol -> UniProt ID 변환"""
    q = f"(gene_exact:{sym}) AND (organism_id:9606) AND (reviewed:true)"
    url = f"{UNIPROT_BASE}/uniprotkb/search?query={urllib.parse.quote(q)}&fields=accession,reviewed,organism_id&format=json&size=1"
    try:
        j = await http.get(url)
        if j.get("results"):
            return {"symbol": sym, "uniprot_id": j["results"][0]["primaryAccession"]}
        # fallback
        q2 = f"(gene:{sym}) AND (organism_id:9606) AND (reviewed:true)"
        url2 = f"{UNIPROT_BASE}/uniprotkb/search?query={urllib.parse.quote(q2)}&fields=accession&format=json&size=1"
        j2 = await http.get(url2)
        if j2.get("results"):
            return {"symbol": sym, "uniprot_id": j2["results"][0]["primaryAccession"]}
    except Exception as e:
        print("UniProt error:", repr(e))
    return {"symbol": sym, "uniprot_id": None}

# Harmonizome: CTD만 사용 + disease_query 매칭
async def fetch_harmonizome(symbol: str, disease_query: Optional[str], match_mode: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
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
            if CTD_TOKEN not in href:
                continue
            pure_term = name.split("/CTD Gene-Disease Associations")[0].strip() or name.strip()
            if disease_query and not _match_text(pure_term, disease_query, match_mode):
                continue
            if href.startswith("/"):
                href = "https://maayanlab.cloud" + href
            out.append({
                "symbol": j.get("symbol", symbol),
                "term": pure_term,
                "href": href,
                "threshold": a.get("thresholdValue"),
                "standardized": a.get("standardizedValue"),
            })
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

# ---------------- DrugProtAI (Playwright) ----------------
async def _scrape_drugprotai(uniprot_id: str) -> Dict[str, Any]:
    """
    DrugProtAI 사이트에서 정보 추출
    1. UniProt ID 입력
    2. Alert 팝업 처리
    3. SHOW DRUGGABILITY INDEX -> XGB, RF 값 추출
    4. EXISTING DRUGS 정보 추출
    """
    global _last_scrape_ts
    
    # 캐시 확인
    ck = _cache_key(f"PW::{uniprot_id}")
    if ck.exists() and (time.time() - ck.stat().st_mtime) < SCRAPER_CACHE_TTL:
        try:
            print(f"✓ Cache hit for {uniprot_id}")
            return json.loads(ck.read_text("utf-8"))
        except Exception:
            pass
    
    # Rate limit
    wait = SCRAPER_MIN_INTERVAL - (time.time() - _last_scrape_ts)
    if wait > 0:
        await asyncio.sleep(wait)
    
    data: Dict[str, Any] = {
        "druggability": {"xgboost": None, "random_forest": None},
        "existing_drugs": []
    }
    
    browser = context = page = None
    try:
        async with async_playwright() as p:
            print(f"→ Scraping {uniprot_id}...")
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1920, "height": 1080}
            )
            page = await context.new_page()
            
            # Alert 자동 수락 핸들러 등록
            page.on("dialog", lambda dialog: asyncio.create_task(dialog.accept()))
            
            # 1. 사이트 접속
            await page.goto(DRUGPROTAI_HOME, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
            
            # 2. UniProt ID 입력
            input_box = page.locator("input[type='text']").first
            await input_box.wait_for(state="visible", timeout=10000)
            await input_box.fill(uniprot_id)
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(4000)
            
            # 3. SHOW DRUGGABILITY INDEX 버튼 클릭
            try:
                show_btn = page.locator("#show-third-section")
                await show_btn.wait_for(state="visible", timeout=10000)
                await show_btn.evaluate("el => el.click()")
                await page.wait_for_timeout(2000)
                print("  ✓ Druggability section opened")
            except Exception as e:
                print(f"  ✗ Failed to open druggability section: {e}")
            
            # 4. XGB 값 읽기 (기본값)
            try:
                di_elem = page.locator("#druggability-index")
                await di_elem.wait_for(state="visible", timeout=5000)
                
                # 애니메이션 안정화 대기 (연속 3번 동일값)
                last_val = None
                stable_count = 0
                for _ in range(12):
                    text = await di_elem.inner_text()
                    text = text.replace("%", "").replace(",", "").strip()
                    try:
                        current = float(text)
                        if last_val is not None and abs(current - last_val) < 0.01:
                            stable_count += 1
                            if stable_count >= 3:
                                data["druggability"]["xgboost"] = current / 100.0
                                print(f"  ✓ XGB: {current}%")
                                break
                        else:
                            stable_count = 0
                        last_val = current
                    except ValueError:
                        pass
                    await page.wait_for_timeout(250)
            except Exception as e:
                print(f"  ✗ XGB read failed: {e}")
            
            # 5. RF 버튼 클릭 후 값 읽기
            try:
                rf_btn = page.locator("#rf-button")
                await rf_btn.wait_for(state="visible", timeout=5000)
                await rf_btn.evaluate("el => el.click()")
                await page.wait_for_timeout(1500)
                
                # RF 값 안정화 대기
                last_val = None
                stable_count = 0
                for _ in range(12):
                    text = await di_elem.inner_text()
                    text = text.replace("%", "").replace(",", "").strip()
                    try:
                        current = float(text)
                        if last_val is not None and abs(current - last_val) < 0.01:
                            stable_count += 1
                            if stable_count >= 3:
                                data["druggability"]["random_forest"] = current / 100.0
                                print(f"  ✓ RF: {current}%")
                                break
                        else:
                            stable_count = 0
                        last_val = current
                    except ValueError:
                        pass
                    await page.wait_for_timeout(250)
            except Exception as e:
                print(f"  ✗ RF read failed: {e}")
            
            # 6. EXISTING DRUGS 버튼 클릭
            try:
                drug_btn = page.locator("#view-drug-info-btn")
                await drug_btn.wait_for(state="visible", timeout=5000)
                await drug_btn.evaluate("el => el.click()")
                await page.wait_for_timeout(2000)
                print("  ✓ Drug info opened")
                
                # 테이블에서 약물 정보 추출
                table = page.locator("table").first
                await table.wait_for(state="visible", timeout=5000)
                rows = await table.locator("tbody tr").all()
                
                for row in rows:
                    cells = await row.locator("td").all_inner_texts()
                    if len(cells) >= 4:
                        # 헤더 행 제외
                        if cells[0].lower().strip() == "drug id":
                            continue
                        
                        data["existing_drugs"].append({
                            "drug_id": cells[0].strip(),
                            "status": cells[1].strip(),
                            "pharmacological_action": cells[2].strip(),
                            "type": cells[3].strip()
                        })
                
                print(f"  ✓ Found {len(data['existing_drugs'])} drugs")
            except Exception as e:
                print(f"  ✗ Drug info failed: {e}")
    
    except Exception as e:
        print(f"✗ Scraping error for {uniprot_id}: {repr(e)}")
    
    finally:
        if context:
            try:
                await context.close()
            except Exception:
                pass
        if browser:
            try:
                await browser.close()
            except Exception:
                pass
    
    _last_scrape_ts = time.time()
    
    # 유효한 결과만 캐시
    if data["druggability"]["xgboost"] or data["druggability"]["random_forest"] or data["existing_drugs"]:
        try:
            ck.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
            print(f"  ✓ Cached")
        except Exception:
            pass
    
    return data

async def fetch_drugprotai(uniprot_id: str) -> Dict[str, Any]:
    """DrugProtAI 데이터 가져오기 (세마포어로 동시 실행 제한)"""
    if not uniprot_id:
        return {}
    
    async with _drugprotai_sem:
        try:
            return await asyncio.wait_for(_scrape_drugprotai(uniprot_id), timeout=DRUGPROTAI_TIMEOUT)
        except asyncio.TimeoutError:
            print(f"✗ Timeout for {uniprot_id}")
            return {}
        except Exception as e:
            print(f"✗ Error for {uniprot_id}: {repr(e)}")
            return {}

# ---------------- routes ----------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """<html><body>
    <h2>Gene List Explorer API (with DrugProtAI)</h2>
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

@app.get("/debug/drugprotai", response_class=JSONResponse)
async def debug_drugprotai(uniprot_id: str):
    data = await fetch_drugprotai(uniprot_id)
    return data

@app.post("/aggregate", response_class=JSONResponse)
async def aggregate(req: AggregateRequest):
    symbols = clean_symbols(req.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="No valid symbols provided")

    print(f"\n{'='*60}")
    print(f"Processing {len(symbols)} symbols")
    print(f"{'='*60}")

    # 1) map symbols -> Ensembl IDs & UniProt IDs in parallel
    ensembl_maps, uniprot_maps = await asyncio.gather(
        asyncio.gather(*[map_symbol_to_ensembl(s) for s in symbols]),
        asyncio.gather(*[fetch_uniprot_id(s) for s in symbols]),
    )

    # 2) per gene tasks (Harmonizome + OpenTargets)
    tasks = []
    for m in ensembl_maps:
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
    for m in ensembl_maps:
        sym = m["symbol"]
        if req.fetch_harmonizome:
            rows, err = results[r_idx]; r_idx += 1
            hz[sym] = rows; hz_err[sym] = err
        if req.fetch_opentargets:
            rows, err = results[r_idx]; r_idx += 1
            ot[sym] = rows; ot_err[sym] = err

    # 3) DrugProtAI (순차 실행)
    dp: Dict[str, Dict[str, Any]] = {}
    dp_err: Dict[str, Optional[str]] = {}
    
    if req.fetch_drugprotai:
        print(f"\nFetching DrugProtAI data...")
        for uniprot_map in uniprot_maps:
            sym = uniprot_map["symbol"]
            uid = uniprot_map.get("uniprot_id")
            
            if not uid:
                dp[sym] = {
                    "uniprot_id": None,
                    "druggability_xgb": None,
                    "druggability_rf": None,
                    "existing_drugs": []
                }
                dp_err[sym] = "No UniProt ID found"
                continue
            
            data = await fetch_drugprotai(uid)
            dg = data.get("druggability", {})
            
            dp[sym] = {
                "uniprot_id": uid,
                "druggability_xgb": dg.get("xgboost"),
                "druggability_rf": dg.get("random_forest"),
                "existing_drugs": data.get("existing_drugs", [])
            }
            dp_err[sym] = None

    # Top-N per gene (OpenTargets)
    topn = max(1, req.topn or 15)
    ot_top: Dict[str, List[Dict[str, Any]]] = {}
    for sym, rows in ot.items():
        rows_sorted = sorted(rows, key=lambda x: (x.get("score") is not None, x.get("score") or -1), reverse=True)
        ot_top[sym] = rows_sorted[:topn]

    print(f"{'='*60}\n")

    return {
        "mappings": ensembl_maps,
        "uniprot_mappings": uniprot_maps,
        "harmonizome": hz,
        "harmonizome_errors": hz_err,
        "opentargets": ot,
        "opentargets_errors": ot_err,
        "opentargets_top": ot_top,
        "drugprotai": dp,
        "drugprotai_errors": dp_err,
    }
