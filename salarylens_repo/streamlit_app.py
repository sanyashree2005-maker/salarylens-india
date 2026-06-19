import os, pickle, warnings, re
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SalaryLens India",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif}
section[data-testid="stSidebar"]{display:none}
#MainMenu,footer,header{visibility:hidden}
.stProgress>div>div{background:linear-gradient(90deg,#6366f1,#8b5cf6)}
div[data-testid="stHorizontalBlock"]{gap:0.8rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_arts():
    with open("/content/models/salary_model_artifacts.pkl","rb") as f:
        return pickle.load(f)

arts        = load_arts()
lgbm_m      = arts["lgbm_models"]
cb_m        = arts.get("catboost_models",{})
label_enc   = arts["label_enc"]
pca         = arts["pca"]
FEATS       = arts["feats"]
CAT_COLS    = arts["cat_cols"]
skill_cols  = arts["skill_cols"]
emb_cols    = arts["emb_cols"]
shap_imp    = arts.get("shap",{})
SEN_LABELS  = arts["sen_labels"]
CITY_TIER   = arts["city_tier"]
TECH_HUBS   = set(arts["tech_hubs"])
CO_TYPE     = arts["co_type"]
TOP_COS     = arts["top_cos"]
TECH_SKILLS = arts["tech_skills"]
YEAR_W      = {2025:1.20,2024:1.10,2023:1.00,2022:0.85,2021:0.70,2020:0.55}

# ── Extended company classification ─────────────────────────────────────────
# CO_TYPE loaded from artifacts covers 87 named companies.
# EXTENDED_CO_TYPE covers additional companies that appear in training data
# but were not in CO_TYPE during training — adding them here improves
# inference routing to the correct company type bucket (no retraining needed).
EXTENDED_CO_TYPE = {
    # IT Services aliases
    "tcs":"it_services","t c s":"it_services",
    "l&t technology services":"it_services","ltts":"it_services",
    "niit technologies":"it_services","mastech":"it_services",
    # Indian Product missing
    "nykaa":"indian_product","browserstack":"indian_product",
    "zerodha":"indian_product","upstox":"indian_product",
    "oyo":"indian_product","cars24":"indian_product",
    "shiprocket":"indian_product","lenskart":"indian_product",
    "makemytrip":"indian_product","mmt":"indian_product",
    "ixigo":"indian_product","rapido":"indian_product",
    "smallcase":"indian_product","juspay":"indian_product",
    "setu":"indian_product","decentro":"indian_product",
    "khatabook":"indian_product","dunzo":"indian_product",
    "spinny":"indian_product","udaan":"indian_product",
    "unacademy":"indian_product","byjus":"indian_product",
    "byju's":"indian_product","fampay":"indian_product",
    "niyo":"indian_product","fi money":"indian_product",
    "stashfin":"indian_product","rupeek":"indian_product",
    "postman":"indian_product","hasura":"indian_product",
    # MNC Product missing
    "netflix":"mnc_product","airbnb":"mnc_product",
    "stripe":"mnc_product","datadog":"mnc_product",
    "cloudflare":"mnc_product","gitlab":"mnc_product",
    "github":"mnc_product","confluent":"mnc_product",
    "elastic":"mnc_product","okta":"mnc_product",
    "pagerduty":"mnc_product","zendesk":"mnc_product",
    "twilio":"mnc_product","grafana":"mnc_product",
    "hashicorp":"mnc_product","mongodb":"mnc_product",
    "snowflake":"mnc_product","databricks":"mnc_product",
    "figma":"mnc_product","notion":"mnc_product",
    "slack":"mnc_product","zoom":"mnc_product",
    "docusign":"mnc_product","hubspot":"mnc_product",
    "jfrog":"mnc_product","fastly":"mnc_product",
    # Finance missing
    "hdfc":"mnc_finance","icici":"mnc_finance",
    "kotak":"mnc_finance","axis bank":"mnc_finance",
    "standard chartered":"mnc_finance",
    "jp morgan":"mnc_finance","wells fargo":"mnc_finance",
    "bank of america":"mnc_finance","boa":"mnc_finance",
    "idfc":"mnc_finance","rbl bank":"mnc_finance",
    "yes bank":"mnc_finance","sbi":"mnc_finance",
    "paytm payments bank":"mnc_finance",
    # Consulting missing
    "booz allen":"consulting","oliver wyman":"consulting",
    "pa consulting":"consulting","ey-parthenon":"consulting",
    "accenture song":"consulting","zs associates":"consulting",
    "mu sigma":"consulting","fractal analytics":"consulting",
}

# Merge: EXTENDED takes priority over CO_TYPE for overlapping keys
_FULL_CO_TYPE = {**CO_TYPE, **EXTENDED_CO_TYPE}

TIER1_MNCS = {
    "google","amazon","microsoft","meta","apple","netflix","nvidia",
    "qualcomm","adobe","salesforce","uber","linkedin","intuit","atlassian",
    "servicenow","workday","paypal","visa","mastercard",
    # Added missing Tier-1s
    "airbnb","stripe","datadog","cloudflare","snowflake","databricks",
}

def is_tier1(name):
    if not isinstance(name,str): return 0
    return int(any(t in name.strip().lower() for t in TIER1_MNCS))

# Smart keyword-based fallback for truly unknown companies
# Catches patterns the explicit list misses
_IT_KEYWORDS    = ["technologies","technology","tech","solutions","systems",
                    "infotech","software services","consulting services","outsourcing"]
_FINANCE_KW     = ["bank","finance","financial","capital","investment","securities",
                    "insurance","asset management","wealth","trading","fintech"]
_PRODUCT_KW     = ["labs","ai","studio","platform","cloud","data","analytics",
                    "intelligence","automation","robotics","networks"]
_STARTUP_SIGNALS= ["startup","ventures","inc","corp","co.","pvt","limited","ltd"]

def _keyword_classify(n):
    """Classify by name keywords when explicit lookup fails."""
    if any(kw in n for kw in _FINANCE_KW):   return "mnc_finance"
    if any(kw in n for kw in _IT_KEYWORDS):   return "it_services"
    if any(kw in n for kw in _PRODUCT_KW):    return "indian_product"
    return "unknown"

CITY_NORM  = {"bangalore":"bengaluru","gurugram":"gurgaon","bombay":"mumbai",
               "madras":"chennai","calcutta":"kolkata","new delhi":"delhi"}
STATE_CITY = {"karnataka":"bengaluru","telangana":"hyderabad","maharashtra":"pune",
               "haryana":"gurgaon","delhi":"delhi","uttar pradesh":"noida",
               "tamil nadu":"chennai","west bengal":"kolkata","gujarat":"ahmedabad"}

ROLE_RULES = [
    (10,r"engineering manager|software manager","Engineering Manager"),
    (20,r"data scientist|ml scientist|research sci","Data Scientist"),
    (20,r"data engineer|etl engineer","Data Engineer"),
    (20,r"data analyst|bi analyst|analytics eng","Data Analyst"),
    (20,r"machine learning engineer|ml engineer|mlops","ML Engineer"),
    (30,r"backend|back.end","Backend Engineer"),
    (30,r"frontend|front.end","Frontend Engineer"),
    (30,r"full.?stack|fullstack","Full Stack Engineer"),
    (30,r"devops|site reliability","DevOps/SRE"),
    (30,r"software development engineer|software engineer|sde|swe|developer","Software Engineer"),
    (40,r"hardware engineer|vlsi","Hardware Engineer"),
    (50,r"product manager|product owner","Product Manager"),
    (80,r"project manager|scrum master","Project Manager"),
    (90,r"solution architect|enterprise architect","Solution Architect"),
    (100,r"recruiter|talent acquisition|hr manager|human resource|people ops|hrbp","HR/Recruiter"),
]
_RC = [(re.compile(p,re.IGNORECASE),r) for _,p,r in sorted(ROLE_RULES,key=lambda x:x[0])]

def norm_role(t):
    for pat,role in _RC:
        if pat.search(str(t or "").lower()): return role
    return "Other"

def ext_sen(title=""):
    t=str(title or "").lower()
    if re.search(r"\bintern\b|\bfresher\b",t): return 0
    if re.search(r"\bjunior\b|\bassociate\b",t): return 1
    if re.search(r"\bprincipal\b",t): return 5
    if re.search(r"\bdirector\b|\bvp\b",t): return 7
    if re.search(r"\bstaff\b|\btech lead\b",t): return 4
    if re.search(r"\bsenior\b|\bsr\.?\b",t): return 3
    if re.search(r"\bmanager\b",t): return 6
    return 2

def ext_city(loc):
    if not isinstance(loc,str): return "unknown"
    loc=re.sub(r",?\s*india\s*$","",loc.strip().lower()).strip()
    cand=re.sub(r"\b[a-z]{2}\b$","",loc.split(",")[0].strip()).strip()
    cand=CITY_NORM.get(cand,cand)
    if not cand or cand in("empty","india","n/a",""):
        for s,c in STATE_CITY.items():
            if s in loc: return c
        return "unknown"
    return STATE_CITY.get(cand,cand)

def class_co(name):
    """3-layer lookup: exact match → substring → keyword heuristic."""
    if not isinstance(name,str): return "unknown"
    n = name.strip().lower()
    if n in _FULL_CO_TYPE: return _FULL_CO_TYPE[n]
    for k,v in _FULL_CO_TYPE.items():
        if k in n: return v
    return _keyword_classify(n)

def enc_co(name):
    """Return specific company encoding if known, else other."""
    if not isinstance(name,str): return "other"
    n = name.strip().lower()
    all_cos = list(CO_TYPE.keys()) + list(EXTENDED_CO_TYPE.keys())
    for tc in all_cos:
        if tc == n or tc in n: return tc
    return "other"

def ext_skills(title):
    text=str(title or "").lower()
    return {"sk_"+s.replace(" ","_").replace(".","").replace("/","_"):int(s in text)
            for s in TECH_SKILLS}

def mono(p25,p50,p75):
    stack=np.sort(np.stack([np.array(p25),np.array(p50),np.array(p75)],axis=1),axis=1)
    return stack[:,0],stack[:,1],stack[:,2]

@st.cache_resource
def get_emb(): return SentenceTransformer("all-MiniLM-L6-v2")


# ── SIMILAR COMPANY SUGGESTIONS ───────────────────────────────────────────────
# When a company is fully unknown, suggest the closest known companies
# based on name similarity + inferred type signals from the query text

_ALL_KNOWN_COS = sorted(set(list(CO_TYPE.keys()) + list(EXTENDED_CO_TYPE.keys())))

def _suggest_similar(company_name: str, n: int = 5) -> list:
    """
    Returns list of (company_name, company_type) tuples most similar
    to the unknown company. Uses two signals:
    1. String token overlap (catches partial name matches)
    2. Inferred category from keyword (suggests same-type companies)
    """
    if not isinstance(company_name, str): return []
    q = company_name.strip().lower()

    # Signal 1: token overlap score
    q_tokens = set(q.replace("-"," ").replace("."," ").split())
    scored = []
    for co in _ALL_KNOWN_COS:
        co_tokens = set(co.replace("-"," ").replace("."," ").split())
        overlap   = len(q_tokens & co_tokens)
        # Bonus for substring match
        substr    = 2 if (q in co or co in q) else 0
        scored.append((overlap + substr, co))

    # Signal 2: infer category and boost same-type companies
    inferred_type = _keyword_classify(q)
    boosted = []
    for score, co in scored:
        co_type = _FULL_CO_TYPE.get(co, "unknown")
        type_bonus = 1 if co_type == inferred_type else 0
        boosted.append((score + type_bonus, co, co_type))

    # Sort by score desc, then alphabetically
    boosted.sort(key=lambda x: (-x[0], x[1]))

    # Return top n with score > 0, otherwise return top n by category
    results = [(co, ct) for _, co, ct in boosted if _ > 0][:n]
    if len(results) < n:
        # Fill with same-category companies if not enough token matches
        same_type = [
            (co, ct) for _, co, ct in boosted
            if ct == inferred_type and (co, ct) not in results
        ]
        results = (results + same_type)[:n]
    if len(results) < n:
        # Final fallback — just top companies
        top_fallback = [
            (co, ct) for _, co, ct in boosted
            if (co, ct) not in results
        ][:n-len(results)]
        results = results + top_fallback

    return results[:n]

# Curated "best known examples" per type — used as fallback suggestions
_TYPE_EXAMPLES = {
    "mnc_product":    ["Google","Amazon","Microsoft","Meta","Adobe"],
    "indian_product": ["Flipkart","Razorpay","Zerodha","Meesho","Groww"],
    "it_services":    ["Infosys","TCS","Wipro","Cognizant","HCL"],
    "mnc_finance":    ["Goldman Sachs","JPMorgan","Morgan Stanley","Citi","HDFC"],
    "consulting":     ["Deloitte","McKinsey","BCG","Accenture","EY"],
    "unknown":        ["Google","Flipkart","Infosys","Deloitte","Goldman Sachs"],
}

# ─────────────────────────────────────────────────────────────────────────
# COMPANY ENRICHMENT — Approach 3 (web) + Approach 5 (Claude API)
# These only fire when class_co() returns "unknown"
# Existing logic is completely untouched
# ─────────────────────────────────────────────────────────────────────────
# ── APPROACH 3: Web search enrichment ─────────────────────────────────────────
# Uses DuckDuckGo Instant Answer API — free, no API key needed
# Only fires when class_co() returns "unknown"
# Result is cached per company name to avoid repeated calls

import urllib.request, urllib.parse, json as _json

@st.cache_data(show_spinner=False, ttl=86400)
def _web_search_classify(company_name: str) -> dict:
    """
    Search DuckDuckGo for company info and extract company type.
    Returns {"type": str, "description": str, "source": "web"} or None.
    """
    try:
        query = urllib.parse.quote(f"{company_name} company India technology")
        url   = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        req   = urllib.request.Request(url, headers={"User-Agent": "SalaryLensIndia/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = _json.loads(resp.read().decode())

        # Combine abstract + heading for analysis
        text = (
            (data.get("AbstractText") or "") + " " +
            (data.get("Heading")       or "") + " " +
            (data.get("Abstract")      or "")
        ).lower().strip()

        if not text or len(text) < 20:
            return None

        # Rule-based extraction from web text
        co_type = None
        tier1   = 0

        # Check for FAANG/Tier-1 signals
        if any(t in text for t in ["nasdaq","nyse","fortune 500","s&p 500","silicon valley"]):
            co_type = "mnc_product"
            tier1   = 1

        if co_type is None:
            # Finance signals
            if any(w in text for w in [
                "bank","investment bank","hedge fund","asset management",
                "financial services","insurance","securities","stock broker",
                "nbfc","non-banking financial"
            ]):
                co_type = "mnc_finance"

        if co_type is None:
            # IT Services signals
            if any(w in text for w in [
                "it services","outsourcing","bpo","kpo","it consulting",
                "managed services","offshore","staffing"
            ]):
                co_type = "it_services"

        if co_type is None:
            # Consulting signals
            if any(w in text for w in [
                "management consulting","strategy consulting","advisory",
                "professional services","big four","big 4"
            ]):
                co_type = "consulting"

        if co_type is None:
            # Indian product startup signals
            if any(w in text for w in [
                "startup","unicorn","series a","series b","series c",
                "venture capital","backed by","founded in","indian startup",
                "fintech","edtech","healthtech","agritech","proptech",
                "saas","software as a service"
            ]):
                co_type = "indian_product"

        if co_type is None:
            # MNC product signals
            if any(w in text for w in [
                "software company","technology company","cloud computing",
                "enterprise software","platform","developer tools",
                "open source","api","developer"
            ]):
                co_type = "indian_product"  # default to indian_product for unknown tech

        if co_type is None:
            return None

        # Get a clean description (first 120 chars)
        desc = (data.get("AbstractText") or data.get("Abstract") or "")[:120]
        if desc:
            desc = desc.strip().rstrip(".") + "."

        return {
            "type":        co_type,
            "description": desc,
            "source":      "web",
            "tier1":       tier1,
        }

    except Exception:
        return None


# ── APPROACH 5: Claude API classification ─────────────────────────────────────
# Uses claude-haiku-3-5 — fast (~0.5s), cheap, accurate for classification
# Only fires when web search returns None or unknown
# Cached per company name

@st.cache_data(show_spinner=False, ttl=86400)
def _claude_classify(company_name: str) -> dict:
    """
    Ask Claude to classify company type using a minimal prompt.
    Returns {"type": str, "description": str, "source": "claude"} or None.
    """
    try:
        import urllib.request, json as _json

        VALID_TYPES = {
            "mnc_product", "indian_product", "it_services",
            "mnc_finance", "consulting", "unknown"
        }

        prompt = (
            f"Classify this company into exactly one category.\n"
            f"Company: {company_name}\n\n"
            f"Categories:\n"
            f"- mnc_product: global tech product company (Google, Stripe, Datadog etc)\n"
            f"- indian_product: Indian startup or product company (Zerodha, Nykaa etc)\n"
            f"- it_services: IT outsourcing/services (TCS, Infosys etc)\n"
            f"- mnc_finance: bank or financial institution\n"
            f"- consulting: management/strategy consulting firm\n"
            f"- unknown: cannot determine\n\n"
            f"Reply with a JSON object only, no explanation:\n"
            f'{{\"type\": \"<category>\", \"description\": \"<one sentence about company>\"}}'
        )

        payload = _json.dumps({
            "model": "claude-haiku-3-5-20241022",
            "max_tokens": 120,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data    = payload,
            headers = {
                "Content-Type":      "application/json",
                "anthropic-version": "2023-06-01",
            },
            method = "POST"
        )

        with urllib.request.urlopen(req, timeout=6) as resp:
            data = _json.loads(resp.read().decode())

        raw = data["content"][0]["text"].strip()
        # Clean markdown fences if present
        raw = raw.replace("```json","").replace("```","").strip()
        parsed = _json.loads(raw)

        co_type = parsed.get("type","unknown").lower().strip()
        desc    = parsed.get("description","")[:120]

        if co_type not in VALID_TYPES:
            co_type = "unknown"

        return {
            "type":        co_type,
            "description": desc,
            "source":      "claude",
            "tier1":       1 if co_type == "mnc_product" and any(
                t in company_name.lower() for t in [
                    "stripe","airbnb","datadog","cloudflare","snowflake","databricks"
                ]) else 0,
        }

    except Exception:
        return None


# ── SMART CLASSIFY — orchestrator ─────────────────────────────────────────────
# Runs all lookup layers in order, stops at first confident result.
# Existing class_co() and enc_co() are called first (unchanged).
# Web search and Claude are only fallbacks for "unknown" companies.

@st.cache_data(show_spinner=False, ttl=3600)
def smart_classify(company_name: str) -> dict:
    """
    Full enrichment pipeline for unknown companies.
    Returns dict with keys: type, enc, tier1, description, source, enriched
    """
    if not isinstance(company_name, str) or not company_name.strip():
        return {"type":"unknown","enc":"other","tier1":0,
                "description":"","source":"none","enriched":False}

    # Layer 1: existing lookup (87 named + 70 extended + keyword)
    co_type = class_co(company_name)
    co_enc  = enc_co(company_name)
    co_t1   = is_tier1(company_name)

    if co_type != "unknown":
        return {"type":co_type,"enc":co_enc,"tier1":co_t1,
                "description":"","source":"lookup","enriched":False}

    # Layer 2: web search (DuckDuckGo, free, no key)
    web_result = _web_search_classify(company_name)
    if web_result and web_result["type"] != "unknown":
        return {
            "type":        web_result["type"],
            "enc":         "other",
            "tier1":       web_result.get("tier1", 0),
            "description": web_result["description"],
            "source":      "web",
            "enriched":    True,
        }

    # Layer 3: Claude API (only if web search found nothing)
    claude_result = _claude_classify(company_name)
    if claude_result and claude_result["type"] != "unknown":
        return {
            "type":        claude_result["type"],
            "enc":         "other",
            "tier1":       claude_result.get("tier1", 0),
            "description": claude_result["description"],
            "source":      "claude",
            "enriched":    True,
        }

    # All layers exhausted — return unknown with description if any
    desc = ""
    if web_result:    desc = web_result.get("description","")
    if claude_result: desc = claude_result.get("description","") or desc

    return {"type":"unknown","enc":"other","tier1":0,
            "description":desc,"source":"none","enriched":False}

# ─────────────────────────────────────────────────────────────────────────
def run_model(job_title,company,yoe_val,city,sen_override=None):
    role=norm_role(job_title)
    si=int(np.clip(sen_override if sen_override is not None else ext_sen(job_title),0,7))
    ct=ext_city(city+", India")
    yoe_f=float(yoe_val); logy=np.log1p(yoe_f)
    eb=int(pd.cut([yoe_f],bins=[-1,2,5,10,15,40],labels=[0,1,2,3,4])[0] or 2)
    skl=ext_skills(job_title)
    # Use smart_classify — runs web+Claude enrichment only for unknown companies
    # For known companies, returns immediately from local lookup (zero latency)
    _cls = smart_classify(company)
    _co_type    = _cls["type"]
    _co_enc     = _cls["enc"]
    _co_tier1   = _cls["tier1"]
    _enriched   = _cls["enriched"]
    _enrich_src = _cls["source"]
    _enrich_desc= _cls["description"]

    row={
        "normalized_role":role,"company_type":_co_type,
        "company_enc":_co_enc,"city_tier":CITY_TIER.get(ct,2),
        "edu_level":2,"seniority_int":si,"yoe":yoe_f,
        "has_yac":0,"is_tech_hub":int(ct in TECH_HUBS),"is_remote":0,
        "is_tier1_mnc":_co_tier1,
        "log_yoe":logy,"exp_bkt":eb,"sen_x_exp":si*logy,
        "yr_feat":YEAR_W.get(2025,1.2),**skl,
    }
    rl=pd.DataFrame([row])
    for col in CAT_COLS:
        le=label_enc[col]; val=str(rl.get(col,pd.Series(["unknown"])).iloc[0])
        val=val if val in set(le.classes_) else le.classes_[0]
        rl[col]=le.transform([val])[0]
    for col in FEATS:
        if col not in rl.columns: rl[col]=0.0
    emb_txt = role + " " + job_title + " " + SEN_LABELS.get(si, "Mid")
    ec=pca.transform(get_emb().encode([emb_txt]).reshape(1,-1))
    for i,col in enumerate(emb_cols): rl[col]=ec[0,i]
    l25=lgbm_m[0.25].predict(rl[FEATS])[0]
    l50=lgbm_m[0.50].predict(rl[FEATS])[0]
    l75=lgbm_m[0.75].predict(rl[FEATS])[0]
    if cb_m:
        rc=pd.DataFrame([row])
        for col in FEATS:
            if col not in rc.columns: rc[col]=0.0
        for i,col in enumerate(emb_cols): rc[col]=ec[0,i]
        l25=0.6*l25+0.4*cb_m[0.25].predict(rc)[0]
        l50=0.6*l50+0.4*cb_m[0.50].predict(rc)[0]
        l75=0.6*l75+0.4*cb_m[0.75].predict(rc)[0]
    a,b,c=mono([l25],[l50],[l75])
    _conf = (
        "high"   if _co_type != "unknown" and role != "Other" else
        "medium" if role != "Other" else
        "low"
    )
    return {
        "p25":round(np.expm1(a[0])/1e5,1),"p50":round(np.expm1(b[0])/1e5,1),
        "p75":round(np.expm1(c[0])/1e5,1),"role":role,
        "seniority":SEN_LABELS.get(si,"Mid"),"si":si,
        "company_type":_co_type,"city":ct,
        "tier":CITY_TIER.get(ct,2),"is_hub":int(ct in TECH_HUBS),
        "conf":_conf,
        # Enrichment metadata — used by UI to show source badge
        "enriched":    _enriched,
        "enrich_src":  _enrich_src,
        "enrich_desc": _enrich_desc,
    }

MATRIX_ROLES = [
    "Data Scientist","ML Engineer","Software Engineer","Data Engineer",
    "Data Analyst","Product Manager","DevOps/SRE",
    "Backend Engineer","Frontend Engineer","Full Stack Engineer",
    "Engineering Manager","Solution Architect",
]
# Matrix display subset — keep table readable (7 roles max)
MATRIX_DISPLAY_ROLES = [
    "Data Scientist","ML Engineer","Software Engineer",
    "Data Engineer","Data Analyst","Product Manager","DevOps/SRE",
]
MATRIX_COS   = [
    ("Google",          "linear-gradient(135deg,#6366f1,#8b5cf6)"),
    ("Amazon",          "linear-gradient(135deg,#f59e0b,#fbbf24)"),
    ("Flipkart",        "linear-gradient(135deg,#0ea5e9,#38bdf8)"),
    ("Infosys",         "linear-gradient(135deg,#22c55e,#4ade80)"),
    ("Deloitte",        "linear-gradient(135deg,#8b5cf6,#a78bfa)"),
    ("Unknown Startup", "linear-gradient(135deg,#ec4899,#f472b6)"),
]

COMPANY_CHOICES = {
    "FAANG / Tier-1 MNC": ["Google","Amazon","Microsoft","Meta","Apple","Netflix","Nvidia","Adobe","Salesforce","LinkedIn"],
    "Indian Product":      ["Flipkart","Meesho","Razorpay","PhonePe","Swiggy","Zomato","CRED","Groww","Paytm","Freshworks"],
    "IT Services":         ["Infosys","TCS","Wipro","HCL","Tech Mahindra","Cognizant","Accenture","LTIMindtree","Mphasis"],
    "MNC Finance":         ["Goldman Sachs","JPMorgan","Morgan Stanley","Citi","Barclays","Deutsche Bank"],
    "Consulting":          ["Deloitte","McKinsey","BCG","Bain","PwC","KPMG","EY","ThoughtWorks"],
}

CO_TYPE_COLORS = {
    "mnc_product":    "#6366f1",
    "indian_product": "#0ea5e9",
    "it_services":    "#22c55e",
    "mnc_finance":    "#f59e0b",
    "consulting":     "#8b5cf6",
    "unknown":        "#94a3b8",
}

def card(lbl,val,bg,caption="",delta=""):
    h = (f'<div style="background:{bg};border-radius:16px;padding:1.5rem;text-align:center;'
         f'box-shadow:0 8px 24px rgba(0,0,0,0.12);margin-bottom:8px">'
         f'<div style="font-size:11px;font-weight:600;letter-spacing:0.1em;'
         f'color:rgba(255,255,255,0.75);text-transform:uppercase;margin-bottom:8px">{lbl}</div>'
         f'<div style="font-size:32px;font-weight:800;color:white;line-height:1">{val}</div>')
    if caption: h += f'<div style="font-size:12px;color:rgba(255,255,255,0.65);margin-top:6px">{caption}</div>'
    if delta:   h += f'<div style="font-size:12px;color:rgba(255,255,255,0.85);margin-top:4px;font-weight:600">{delta}</div>'
    h += '</div>'
    return h

def mini_card(lbl,val,bg):
    return (f'<div style="background:{bg};border-radius:12px;padding:1rem;text-align:center;margin-bottom:6px">'
            f'<div style="font-size:10px;color:rgba(255,255,255,0.7);text-transform:uppercase;letter-spacing:0.07em">{lbl}</div>'
            f'<div style="font-size:22px;font-weight:800;color:white;margin-top:4px">{val}</div></div>')

def pill(txt,bg,fg):
    return (f'<span style="background:{bg};color:{fg};padding:5px 14px;border-radius:999px;'
            f'font-size:13px;font-weight:600;margin:3px;display:inline-block">{txt}</span>')

def sig_card(lbl,val,bg):
    return (f'<div style="background:{bg};border-radius:10px;padding:12px 8px;text-align:center">'
            f'<div style="font-size:9px;color:rgba(255,255,255,0.7);text-transform:uppercase;letter-spacing:0.08em">{lbl}</div>'
            f'<div style="font-size:13px;font-weight:700;color:white;margin-top:3px">{val}</div></div>')

def row_item(lbl,val,vc):
    return (f'<div style="display:flex;justify-content:space-between;padding:9px 14px;'
            f'background:#f8fafc;border-radius:8px;margin-bottom:6px;border:1px solid #e2e8f0">'
            f'<span style="font-weight:500;color:#1e293b">{lbl}</span>'
            f'<span style="color:{vc};font-weight:700">{val}</span></div>')

def tip_box(text):
    return (f'<div style="background:#f0fdf4;border-left:4px solid #22c55e;border-radius:8px;'
            f'padding:12px 16px;margin:8px 0;font-size:14px;color:#15803d;font-weight:500">💡 {text}</div>')

def alert_box(text,bg,bd,fg):
    return (f'<div style="background:{bg};border-left:4px solid {bd};border-radius:8px;'
            f'padding:12px 16px;margin:12px 0;font-size:14px;color:{fg};font-weight:500">{text}</div>')

def range_bar(co_name,p25,p50,p75,max_val,co_color):
    bar_w  = int((p75-p25)/max_val*100)
    bar_s  = int(p25/max_val*100)
    mid_p  = int((p50-p25)/(p75-p25+0.01)*bar_w)
    return (f'<div style="margin-bottom:14px">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
            f'<span style="font-size:13px;font-weight:600;color:#1e293b">{co_name}</span>'
            f'<span style="font-size:13px;color:{co_color};font-weight:700">₹{p50}L '
            f'<span style="color:#94a3b8;font-weight:400;font-size:11px">(₹{p25}–{p75}L)</span></span></div>'
            f'<div style="background:#f1f5f9;border-radius:999px;height:10px;position:relative">'
            f'<div style="position:absolute;left:{bar_s}%;width:{bar_w}%;height:100%;'
            f'background:{co_color};opacity:0.25;border-radius:999px"></div>'
            f'<div style="position:absolute;left:{bar_s+mid_p//4}%;width:3px;height:100%;'
            f'background:{co_color};border-radius:999px"></div></div></div>')

def heat_color(val,vmin,vmax):
    ratio=(val-vmin)/(vmax-vmin+0.01)
    if ratio>0.75:   return "#dcfce7","#15803d"
    elif ratio>0.50: return "#fef9c3","#a16207"
    elif ratio>0.25: return "#fff7ed","#c2410c"
    else:            return "#fef2f2","#b91c1c"

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 0.5rem">
  <div style="font-size:36px;font-weight:800;color:#0f172a">💰 SalaryLens India</div>
  <div style="font-size:15px;color:#64748b;margin-top:6px">
    Real salary intelligence — built on
    <b style="color:#6366f1">177K+ verified data points</b>
    from Levels.fyi · Glassdoor · Job Postings
  </div>
</div>
""", unsafe_allow_html=True)

sc1,sc2,sc3,sc4 = st.columns(4)
for col,lbl,val,bg in [
    (sc1,"Data Points","177K+","linear-gradient(135deg,#6366f1,#8b5cf6)"),
    (sc2,"Cities","11","linear-gradient(135deg,#0ea5e9,#38bdf8)"),
    (sc3,"Companies","200+","linear-gradient(135deg,#22c55e,#4ade80)"),
    (sc4,"Roles","18","linear-gradient(135deg,#f59e0b,#fbbf24)"),
]:
    col.markdown(card(lbl,val,bg),unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:2px solid #f1f5f9'>",unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs([
    "🔍  Predict Salary",
    "🏢  Company Compare",
    "📊  Salary Band Matrix",
    "🌆  City Compare",
])

# ── TAB 1: PREDICT ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Enter Your Details")
    r1c1,r1c2,r1c3 = st.columns(3)
    with r1c1: job_title   = st.text_input("🎯 Job Title",  placeholder="e.g. Senior Data Scientist")
    with r1c2: company     = st.text_input("🏢 Company",    placeholder="e.g. Google, Flipkart")
    with r1c3: city        = st.selectbox("📍 City",["Bengaluru","Hyderabad","Pune","Gurgaon",
                                "Noida","Mumbai","Chennai","Delhi","Ahmedabad","Kolkata","Kochi"])
    r2c1,r2c2,r2c3 = st.columns(3)
    with r2c1: yoe         = st.number_input("💼 Years of Experience",0,30,3)
    with r2c2: current_sal = st.number_input("💰 Current CTC (LPA) — optional",0.0,200.0,0.0,0.5)
    with r2c3:
        sen_opts = {v:k for k,v in SEN_LABELS.items()}
        sen_sel  = st.selectbox("🎓 Seniority",["Auto-detect"]+list(SEN_LABELS.values()))
        sen_ov   = None if sen_sel=="Auto-detect" else sen_opts[sen_sel]
    predict_btn = st.button("🔍  Get Salary Range",use_container_width=True,type="primary",key="pred_btn")

    if predict_btn:
        if not job_title:
            st.warning("Please enter a Job Title.")
            st.stop()

        # ── UNKNOWN COMPANY GATE ──────────────────────────────────────────────
        # Before running the model, check if company is known.
        # If completely unknown after all enrichment layers, stop and show
        # a clear "no data" message with similar company suggestions.
        # This only applies when the user actually typed a company name.
        _co_input = (company or "").strip()
        if _co_input:
            with st.spinner(f"Looking up {_co_input}..."):
                _cls_check = smart_classify(_co_input)

            if _cls_check["type"] == "unknown":
                # Company is fully unknown — hard stop, no prediction
                st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1rem 0'>",unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#fef2f2;border:1.5px solid #fca5a5;border-radius:14px;' +
                    f'padding:1.4rem 1.6rem;margin-bottom:1rem">' +
                    f'<div style="font-size:18px;font-weight:700;color:#b91c1c;margin-bottom:6px">' +
                    f'❌  No data available for &quot;{_co_input}&quot;</div>' +
                    f'<div style="font-size:14px;color:#7f1d1d;line-height:1.7">' +
                    f'This company is not in our dataset and could not be classified ' +
                    f'by web search or AI. We cannot make a reliable salary prediction ' +
                    f'without knowing the company type — predictions would be meaningless.</div>' +
                    f'</div>',
                    unsafe_allow_html=True)

                # Similar company suggestions
                _suggestions = _suggest_similar(_co_input, n=5)
                if _suggestions:
                    st.markdown("#### 💡 Try one of these similar companies instead:")
                    sug_cols = st.columns(len(_suggestions))
                    for col, (sug_co, sug_type) in zip(sug_cols, _suggestions):
                        sug_label = sug_type.replace("_"," ").title()
                        sug_color = CO_TYPE_COLORS.get(sug_type, "#94a3b8")
                        col.markdown(
                            f'<div style="border:1.5px solid {sug_color};border-radius:12px;' +
                            f'padding:12px;text-align:center;cursor:pointer">' +
                            f'<div style="font-size:14px;font-weight:700;color:#0f172a">{sug_co}</div>' +
                            f'<div style="font-size:11px;color:{sug_color};margin-top:4px;font-weight:600">{sug_label}</div>' +
                            f'</div>',
                            unsafe_allow_html=True)

                # Also show what we do know for the role + city (no company)
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:12px;' +
                    f'padding:1rem 1.4rem;font-size:14px;color:#0369a1">' +
                    f'💡 <b>Alternatively:</b> leave the Company field empty to get the ' +
                    f'<b>market average salary</b> for {job_title} in {city} without ' +
                    f'company-specific adjustment.</div>',
                    unsafe_allow_html=True)
                st.stop()

        # Company is known (or no company entered) — proceed with prediction
        with st.spinner("Analysing 177K+ data points..."):
            r = run_model(job_title,company or "unknown",yoe,city,sen_ov)
        p25,p50,p75 = r["p25"],r["p50"],r["p75"]
        span        = round(p75-p25,1)
        co_label    = r["company_type"].replace("_"," ").title()
        st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1rem 0'>",unsafe_allow_html=True)
        # Build pill row — same as before + enrichment badge if company was enriched
        _conf_pill = pill(
            "✅ High Confidence" if r["conf"]=="high" else "⚠️ Medium Confidence",
            "#f0fdf4" if r["conf"]=="high" else "#fef9c3",
            "#15803d" if r["conf"]=="high" else "#a16207"
        )
        _enrich_pill = ""
        if r.get("enriched"):
            _src = r.get("enrich_src","")
            if _src == "web":
                _enrich_pill = pill("🌐 Company info via web search","#e0f2fe","#0369a1")
            elif _src == "claude":
                _enrich_pill = pill("🤖 Company classified by AI","#ede9fe","#5b21b6")
        ph=(pill(f"🎯 {r['role']}","#ede9fe","#5b21b6")+
            pill(f"📊 {r['seniority']}","#dcfce7","#15803d")+
            pill(f"🏢 {co_label}","#fff7ed","#c2410c")+
            pill(f"📍 {r['city'].title()} T{r['tier']}","#f0f9ff","#0369a1")+
            _conf_pill + _enrich_pill)
        st.markdown(ph,unsafe_allow_html=True)
        # Show enrichment description if available
        if r.get("enrich_desc"):
            st.markdown(
                f'<div style="background:#f8fafc;border-left:3px solid #cbd5e1;'
                f'border-radius:0 6px 6px 0;padding:8px 14px;margin:4px 0 8px;'
                f'font-size:12px;color:#64748b">📖 {r["enrich_desc"]}</div>',
                unsafe_allow_html=True)
        st.markdown("")
        c1,c2,c3=st.columns(3)
        c1.markdown(card("P25 — Floor",   f"₹{p25} LPA","linear-gradient(135deg,#3b82f6,#60a5fa)","25% earn below this"),unsafe_allow_html=True)
        c2.markdown(card("P50 — Expected",f"₹{p50} LPA","linear-gradient(135deg,#22c55e,#4ade80)","Market median",f"+₹{round(p50-p25,1)}L above P25"),unsafe_allow_html=True)
        c3.markdown(card("P75 — Ceiling", f"₹{p75} LPA","linear-gradient(135deg,#f59e0b,#fbbf24)","Top 25% of earners",f"+₹{round(p75-p50,1)}L above P50"),unsafe_allow_html=True)
        st.progress((p50-p25)/(span+0.01),text=f"P50 ₹{p50}L at {round((p50-p25)/(span+0.01)*100)}% of range  |  ₹{p25}L to ₹{p75}L  |  Span ₹{span}L")
        st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1.5rem 0'>",unsafe_allow_html=True)

        if current_sal > 0:
            gap=round(p50-current_sal,1)
            st.markdown("### 📊 Are You Underpaid?")
            uc1,uc2,uc3=st.columns(3)
            uc1.markdown(card("Your Current CTC",f"₹{current_sal}L","linear-gradient(135deg,#64748b,#94a3b8)"),unsafe_allow_html=True)
            uc2.markdown(card("Market Median",   f"₹{p50}L",       "linear-gradient(135deg,#6366f1,#8b5cf6)"),unsafe_allow_html=True)
            uc3.markdown(card("Gap vs Market",
                f"{'−' if gap>0 else '+'}₹{abs(gap)}L",
                "linear-gradient(135deg,#ef4444,#f87171)" if gap>2 else
                "linear-gradient(135deg,#f59e0b,#fbbf24)" if gap>0 else
                "linear-gradient(135deg,#22c55e,#4ade80)",
                "Underpaid" if gap>2 else "Slightly below" if gap>0 else "Well compensated"),
                unsafe_allow_html=True)
            if gap>2:   st.markdown(alert_box(f"⚠️ ₹{gap}L below market median. Strong case for a raise or switch.","#fef2f2","#ef4444","#b91c1c"),unsafe_allow_html=True)
            elif gap>0: st.markdown(alert_box(f"⚡ Slightly below by ₹{gap}L. Raise at next review.","#fef9c3","#eab308","#92400e"),unsafe_allow_html=True)
            else:       st.markdown(alert_box(f"✅ ₹{abs(gap)}L above market median. Well positioned.","#f0fdf4","#22c55e","#15803d"),unsafe_allow_html=True)
            st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1.5rem 0'>",unsafe_allow_html=True)

        st.markdown(f"### 📈 Salary Growth Curve — {r['role']}")
        st.caption(f"{r['city'].title()} · {co_label}")
        pts=[run_model(job_title,company or "unknown",yr,city)["p50"] for yr in range(0,16)]
        st.line_chart(pd.DataFrame({"Experience (Years)":range(0,16),"Expected Salary (LPA)":pts}).set_index("Experience (Years)"))
        st.caption(f"At {yoe} yrs: ₹{p50}L  ·  At 5: ₹{pts[5]}L  ·  At 10: ₹{pts[10]}L  ·  At 15: ₹{pts[15]}L")
        st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1.5rem 0'>",unsafe_allow_html=True)

        st.markdown("### 🔑 Key Signals")
        sg=st.columns(6)
        for col,lbl,val,bg in [
            (sg[0],"Role",      r["role"],                      "linear-gradient(135deg,#6366f1,#8b5cf6)"),
            (sg[1],"Seniority", r["seniority"],                 "linear-gradient(135deg,#0ea5e9,#38bdf8)"),
            (sg[2],"Company",   co_label,                       "linear-gradient(135deg,#f59e0b,#fbbf24)"),
            (sg[3],"City Tier", f"Tier {r['tier']}",            "linear-gradient(135deg,#22c55e,#4ade80)"),
            (sg[4],"Experience",f"{yoe} yrs",                   "linear-gradient(135deg,#8b5cf6,#a78bfa)"),
            (sg[5],"Tech Hub",  "Yes ✅" if r["is_hub"] else "No","linear-gradient(135deg,#ec4899,#f472b6)"),
        ]:
            col.markdown(sig_card(lbl,val,bg),unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1.5rem 0'>",unsafe_allow_html=True)

        if shap_imp:
            st.markdown("### 🧠 What Drove This Prediction")
            top6=sorted(shap_imp.items(),key=lambda x:-x[1])[:6]
            max_v=top6[0][1]
            for feat,val in top6:
                clean=feat.replace("sk_","Skill: ").replace("_"," ").title()
                pct=int(val/max_v*100)
                st.markdown(
                    f'<div style="margin-bottom:12px">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:5px">'
                    f'<span style="font-size:13px;font-weight:600;color:#374151">{clean}</span>'
                    f'<span style="font-size:13px;color:#6366f1;font-weight:700">{pct}%</span></div>'
                    f'<div style="background:#e5e7eb;border-radius:999px;height:8px">'
                    f'<div style="background:linear-gradient(90deg,#6366f1,#8b5cf6);width:{pct}%;height:100%;border-radius:999px"></div></div></div>',
                    unsafe_allow_html=True)
            st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1.5rem 0'>",unsafe_allow_html=True)

        st.markdown("### 💼 Negotiation Intelligence")
        co2=r["company_type"]; si2=r["si"]
        for t in [
            {"it_services":"IT services firms have rigid bands — use Levels.fyi data to justify above-band offers.",
             "mnc_product":"MNC product companies have signing bonus flexibility — negotiate that separately from base.",
             "indian_product":"Indian product startups offer ESOPs — push for accelerated vesting if base is capped.",
             "consulting":"Consulting firms move in cohorts — focus on joining bonus and early grade acceleration."
            }.get(co2,"Anchor on market P75 and let them negotiate down. Never reveal your current salary first."),
            f"{r['city'].title()} is a top tech hub — demand the hub premium on top of base." if r["is_hub"]
            else "Non-metro: negotiate remote work option or explicit relocation allowance.",
            "At Staff/Lead+ level: total comp (ESOPs + bonus) matters more than base." if si2>=4
            else "Early career: prioritise fast promotion clause, learning budget, and 6-month review."
        ]:
            st.markdown(tip_box(t),unsafe_allow_html=True)

    else:
        st.markdown("""
> **How to use:** Enter your job title, company, city and years of experience on the left,
> then click **Get Salary Range**. The model returns a P25 / P50 / P75 salary band —
> the floor, expected median, and top-of-range for your profile.
> Optionally enter your current CTC to see if you are underpaid vs the market.
""")
        st.markdown("## 🌏 Indian Tech Salary Landscape")
        mi1,mi2,mi3=st.columns(3)
        for col,title,rows in [
            (mi1,"🏆 Highest Paying Roles",
             [("ML Engineer","₹42L","#6366f1"),("Product Manager","₹40L","#6366f1"),
              ("Data Scientist","₹38L","#6366f1"),("DevOps/SRE","₹35L","#6366f1"),
              ("Software Engineer","₹32L","#6366f1")]),
            (mi2,"📍 Highest Paying Cities",
             [("Bengaluru","₹38L","#0ea5e9"),("Mumbai","₹36L","#0ea5e9"),
              ("Hyderabad","₹34L","#0ea5e9"),("Gurgaon","₹33L","#0ea5e9"),
              ("Pune","₹30L","#0ea5e9")]),
            (mi3,"🏢 Company Type Premium",
             [("MNC Product","+85%","#22c55e"),("MNC Finance","+78%","#22c55e"),
              ("Indian Product","+45%","#22c55e"),("Consulting","+30%","#22c55e"),
              ("IT Services","Baseline","#64748b")]),
        ]:
            with col:
                st.markdown(f"#### {title}")
                for lbl,val,vc in rows:
                    st.markdown(row_item(lbl,val,vc),unsafe_allow_html=True)

# ── TAB 2: COMPANY COMPARE ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### Compare Salary Ranges Across Companies")
    st.markdown("""
> **How it works:** Select a role, city and experience level. Pick from 40+ known companies
> using the checkboxes below — or type any company name in the **Custom** box.
> The model predicts P25 / P50 / P75 salary ranges for each company based on its type,
> location tier and FAANG classification. Results are ranked by median salary.
""")

    cc1,cc2,cc3=st.columns(3)
    with cc1: cmp_role=st.selectbox("🎯 Role",MATRIX_ROLES,key="cmp_role")
    with cc2: cmp_city=st.selectbox("📍 City",["Bengaluru","Hyderabad","Pune","Gurgaon","Noida","Mumbai","Chennai","Delhi","Ahmedabad","Kolkata","Kochi","Jaipur","Chandigarh","Coimbatore","Indore","Nagpur","Lucknow"],key="cmp_city")
    with cc3: cmp_yoe =st.number_input("💼 Years of Experience",0,30,5,key="cmp_yoe")

    # ── Quick-select known companies by group ─────────────────────────────────
    st.markdown("**Quick-select known companies:**")
    selected_cos = []
    for grp, cos in COMPANY_CHOICES.items():
        with st.expander(grp, expanded=(grp=="FAANG / Tier-1 MNC")):
            grp_cols = st.columns(5)
            for i, co in enumerate(cos):
                with grp_cols[i % 5]:
                    if st.checkbox(co, key=f"chk_{co}",
                                   value=co in ["Google","Amazon","Flipkart","Infosys","Deloitte"]):
                        selected_cos.append(co)

    # ── Custom company input ──────────────────────────────────────────────────
    st.markdown("**Or add any other company:**")
    custom_input = st.text_input(
        "Type company names separated by commas",
        placeholder="e.g. Zeta, Juspay, Nykaa, Postman",
        key="custom_cos"
    )
    if custom_input.strip():
        custom_list = [c.strip() for c in custom_input.split(",") if c.strip()]
        if custom_list:
            st.caption(f"Will also compare: {', '.join(custom_list)}")
            selected_cos = selected_cos + custom_list

    if selected_cos:
        st.caption(f"**{len(selected_cos)} companies selected** — results ranked by P50 median")
    cmp_btn=st.button("Compare Companies →",type="primary",use_container_width=True,key="cmp_btn")

    if cmp_btn and selected_cos:
        if len(selected_cos)>8:
            st.warning("Select 8 or fewer companies for a clean comparison.")
        else:
            with st.spinner(f"Computing salary ranges for {len(selected_cos)} companies..."):
                results=[]
                for co in selected_cos:
                    r=run_model(cmp_role,co,cmp_yoe,cmp_city)
                    results.append({"company":co,"co_type":r["company_type"],
                                    "p25":r["p25"],"p50":r["p50"],"p75":r["p75"],
                                    "color":CO_TYPE_COLORS.get(r["company_type"],"#94a3b8")})
                results.sort(key=lambda x:-x["p50"])
                best=results[0]; max_val=results[0]["p75"]*1.1

            st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1rem 0'>",unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:14px;padding:1.2rem 1.8rem;margin-bottom:1.5rem;display:flex;justify-content:space-between;align-items:center">'
                f'<div><div style="font-size:12px;color:rgba(255,255,255,0.7);font-weight:600;text-transform:uppercase;letter-spacing:0.08em">🏆 Highest Paying — {cmp_role} · {cmp_yoe} yrs · {cmp_city}</div>'
                f'<div style="font-size:24px;font-weight:800;color:white;margin-top:4px">{best["company"]}</div></div>'
                f'<div style="text-align:right"><div style="font-size:32px;font-weight:800;color:white">₹{best["p50"]}L</div>'
                f'<div style="font-size:12px;color:rgba(255,255,255,0.7)">P50 · Range ₹{best["p25"]}–{best["p75"]}L</div></div></div>',
                unsafe_allow_html=True)

            st.markdown("#### Salary Range Comparison")
            bar_html=""
            for res in results:
                bar_html+=range_bar(res["company"],res["p25"],res["p50"],res["p75"],max_val,res["color"])
            st.markdown(bar_html,unsafe_allow_html=True)
            st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1.5rem 0'>",unsafe_allow_html=True)

            st.markdown("#### Detailed Breakdown")
            n_cols=min(4,len(results))
            for i in range(0,len(results),n_cols):
                batch=results[i:i+n_cols]
                cols=st.columns(len(batch))
                for col,res in zip(cols,batch):
                    rank=results.index(res)+1
                    medal={1:"🥇",2:"🥈",3:"🥉"}.get(rank,f"#{rank}")
                    co_label=res["co_type"].replace("_"," ").title()
                    col.markdown(
                        f'<div style="border:2px solid {res["color"]};border-radius:14px;padding:1.2rem;margin-bottom:8px">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:10px">'
                        f'<span style="font-size:15px;font-weight:700;color:#0f172a">{res["company"]}</span>'
                        f'<span style="font-size:18px">{medal}</span></div>'
                        f'<div style="font-size:11px;color:#94a3b8;margin-bottom:8px">{co_label}</div>'
                        f'<div style="background:#f8fafc;border-radius:8px;padding:10px">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                        f'<span style="font-size:12px;color:#64748b">P25 Floor</span>'
                        f'<span style="font-size:13px;font-weight:600;color:#3b82f6">₹{res["p25"]}L</span></div>'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                        f'<span style="font-size:12px;color:#64748b">P50 Target</span>'
                        f'<span style="font-size:15px;font-weight:700;color:{res["color"]}">₹{res["p50"]}L</span></div>'
                        f'<div style="display:flex;justify-content:space-between">'
                        f'<span style="font-size:12px;color:#64748b">P75 Ceiling</span>'
                        f'<span style="font-size:13px;font-weight:600;color:#f59e0b">₹{res["p75"]}L</span></div></div>'
                        + (f'<div style="margin-top:8px;font-size:12px;color:#22c55e;font-weight:600">+₹{round(res["p50"]-results[-1]["p50"],1)}L vs lowest</div>' if rank==1 else "")
                        + '</div>',
                        unsafe_allow_html=True)

            st.markdown("#### Summary Table")
            df_tbl=pd.DataFrame([{
                "Rank":    str(results.index(r)+1),
                "Company": r["company"],
                "Type":    r["co_type"].replace("_"," ").title(),
                "P25":     f"₹{r['p25']}L",
                "P50":     f"₹{r['p50']}L",
                "P75":     f"₹{r['p75']}L",
                "Range":   f"₹{round(r['p75']-r['p25'],1)}L span",
                "vs Best": f"-₹{round(results[0]['p50']-r['p50'],1)}L" if r!=results[0] else "🏆 Best",
            } for r in results])
            st.dataframe(df_tbl,use_container_width=True,hide_index=True)
    elif cmp_btn:
        st.info("Select at least one company.")
    else:
        st.info("Select companies above and click Compare Companies →")

# ── TAB 3: SALARY BAND MATRIX ───────────────────────────────────────────────────
with tab3:
    st.markdown("### Salary Band Matrix — P50 by Role × Company")
    st.markdown("""
> **How it works:** This matrix shows the **model-predicted P50 (median) salary** for every
> combination of role and company. It uses the same ML model as the Predict tab — so values
> reflect how the model learned company type (FAANG vs IT Services vs Consulting),
> city tier, and experience level, not raw averages from data.
>
> **Color coding:** 🟢 Green = top quartile · 🟡 Yellow = above median ·
> 🟠 Orange = below median · 🔴 Red = bottom quartile — all relative to values in this matrix.
>
> ⚠️ **Note:** High-end companies (Google, Amazon) may be slightly underestimated at senior levels
> due to data sparsity above ₹70L in the training set. Use as a directional benchmark.
""")
    mx1,mx2=st.columns(2)
    with mx1: mx_city=st.selectbox("📍 City",["Bengaluru","Hyderabad","Pune","Gurgaon","Noida","Mumbai","Chennai","Delhi","Ahmedabad","Kolkata","Kochi","Jaipur","Chandigarh","Coimbatore","Indore","Nagpur","Lucknow"],key="mx_city")
    with mx2: mx_yoe =st.number_input("💼 Years of Experience",0,30,5,key="mx_yoe")
    gen_btn=st.button("Generate Matrix →",type="primary",use_container_width=True,key="gen_btn")

    if gen_btn:
        matrix_cos=[co for co,_ in MATRIX_COS]
        with st.spinner("Computing salary matrix..."):
            rows=[]
            for role in MATRIX_DISPLAY_ROLES:
                row={"Role":role}
                for co in matrix_cos:
                    r=run_model(role,co,mx_yoe,mx_city)
                    row[co]=r["p50"]
                rows.append(row)
            df_matrix=pd.DataFrame(rows).set_index("Role")

        st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1rem 0'>",unsafe_allow_html=True)
        st.markdown(f"#### P50 Salary (LPA) — {mx_city} · {mx_yoe} Years Experience")

        all_vals=df_matrix.values.flatten()
        vmin,vmax=all_vals.min(),all_vals.max()

        # Build HTML table — no font-family quotes issue
        tbl  = '<table style="width:100%;border-collapse:separate;border-spacing:4px">'
        tbl += '<tr><th style="text-align:left;padding:8px 12px;font-size:12px;color:#94a3b8;font-weight:600">ROLE</th>'
        for co,bg in MATRIX_COS:
            tbl += f'<th style="padding:8px 12px;border-radius:8px;background:{bg};font-size:12px;color:white;font-weight:700;text-align:center">{co}</th>'
        tbl += '</tr>'
        for role in MATRIX_DISPLAY_ROLES:
            tbl += f'<tr><td style="padding:10px 12px;font-size:13px;font-weight:600;color:#374151;white-space:nowrap">{role}</td>'
            for co,_ in MATRIX_COS:
                val=df_matrix.loc[role,co]
                bg,fg=heat_color(val,vmin,vmax)
                tbl += f'<td style="padding:10px 12px;text-align:center;border-radius:8px;background:{bg};color:{fg};font-size:14px;font-weight:700">₹{val}L</td>'
            tbl += '</tr>'
        tbl += '</table>'
        st.markdown(tbl,unsafe_allow_html=True)
        st.markdown("")

        best_co  =df_matrix.mean().idxmax()
        best_role=df_matrix[matrix_cos[0]].idxmax()
        gap_role =(df_matrix.max(axis=1)-df_matrix.min(axis=1)).idxmax()
        gap_val  =round((df_matrix.max(axis=1)-df_matrix.min(axis=1)).max(),1)

        i1,i2,i3=st.columns(3)
        i1.markdown(card("Highest Paying Company",best_co,"linear-gradient(135deg,#6366f1,#8b5cf6)",
                         f"Avg ₹{df_matrix[best_co].mean():.1f}L across roles"),unsafe_allow_html=True)
        i2.markdown(card(f"Top Role at {matrix_cos[0]}",best_role,"linear-gradient(135deg,#22c55e,#4ade80)",
                         f"₹{df_matrix.loc[best_role,matrix_cos[0]]}L median"),unsafe_allow_html=True)
        i3.markdown(card("Biggest Company Gap",gap_role,"linear-gradient(135deg,#f59e0b,#fbbf24)",
                         f"₹{gap_val}L difference across companies"),unsafe_allow_html=True)

        csv=df_matrix.to_csv()
        st.download_button("⬇️  Download Matrix as CSV",data=csv,
                           file_name=f"salary_matrix_{mx_city}_{mx_yoe}yrs.csv",mime="text/csv")
    else:
        st.info("Select city + experience and click Generate Matrix →")

# ── TAB 4: CITY COMPARE ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Compare Same Role Across Cities")
    ct1,ct2,ct3=st.columns(3)
    with ct1: cty_title  =st.text_input("Role",placeholder="Data Scientist",key="ct_t")
    with ct2: cty_company=st.text_input("Company",placeholder="Amazon",key="ct_c")
    with ct3: cty_yoe    =st.number_input("Experience (years)",0,30,3,key="ct_y")
    cty_cities=st.multiselect("Select cities",
                   ["Bengaluru","Hyderabad","Pune","Gurgaon","Noida","Mumbai","Chennai","Delhi"],
                   default=["Bengaluru","Hyderabad","Pune"])
    cty_btn=st.button("Compare Cities →",type="primary",use_container_width=True,key="cty_btn")

    if cty_btn and cty_title and len(cty_cities)>=2:
        results=[(c,run_model(cty_title,cty_company or "unknown",cty_yoe,c)) for c in cty_cities]
        best=max(results,key=lambda x:x[1]["p50"])[0]
        cols=st.columns(len(results))
        for i,(city_c,r) in enumerate(results):
            with cols[i]:
                crown=" 🏆 Best Pay" if city_c==best else ""
                st.markdown(f"#### 📍 {city_c}{crown}")
                for lbl,val,bg in [
                    ("P25 Floor",   f"₹{r['p25']}L","#3b82f6"),
                    ("P50 Expected",f"₹{r['p50']}L","#22c55e"),
                    ("P75 Ceiling", f"₹{r['p75']}L","#f59e0b"),
                ]:
                    st.markdown(mini_card(lbl,val,bg),unsafe_allow_html=True)
                if i>0:
                    diff=round(r["p50"]-results[0][1]["p50"],1)
                    sign="+" if diff>=0 else ""
                    color="#22c55e" if diff>=0 else "#ef4444"
                    st.markdown(f'<div style="text-align:center;color:{color};font-weight:600;font-size:14px;margin-top:4px">{sign}₹{diff}L vs {results[0][0]}</div>',unsafe_allow_html=True)
    elif cty_btn:
        st.info("Enter a role and select at least 2 cities.")

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:2rem 0'>",unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#94a3b8;font-size:12px;padding-bottom:1rem">SalaryLens India &nbsp;·&nbsp; Levels.fyi · Glassdoor · Job Postings &nbsp;·&nbsp; For reference only</div>',unsafe_allow_html=True)
