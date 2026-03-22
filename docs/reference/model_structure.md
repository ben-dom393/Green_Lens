# ESG Report 綠漂偵測系統專案架構設計

## 問題定義與七類綠漂的可操作化

你列出的七類分類，對應到 entity["organization","TerraChoice","canadian marketing firm"] 提出的「Seven Sins of Greenwashing」框架，且每一類都有可落地的判準（偏向 B2C 產品宣稱，但同樣能轉化用在 ESG report 的敘述與績效宣稱）。citeturn10view0turn9view0  
系統設計上建議把「綠漂偵測」拆成三層任務，讓輸出可以穩定落在「段落原文 + 判決 + 理由 + 證據」：

第一層是「是否為環境/永續宣稱（green claim）」。第二層是「七類綠漂的多標籤分類（multi-label）」。第三層是「證據鏈（evidence chain）與需要補證據的缺口（missing evidence requests）」。這種拆解和近年的綠漂 NLP 研究總結一致：直接做“綠漂/非綠漂”二元分類通常缺乏可公開、可一致標註的資料集，因此工程上要靠多個子任務組合起來，並用監管文本/準則當作判準來源。citeturn13view0turn4search1turn15view2

七類在系統內的「可判斷訊號」可以清楚定義（這部分直接決定你後面模組怎麼寫）：

- Vague claims：段落出現「綠色、環保、永續、友善、eco-friendly」等總括性語彙，缺少可檢驗的範圍、方法、數據、基準線；監管與指南文件普遍要求此類主張需可被具體限定與實證支持。citeturn10view0turn11view0turn15view5turn15view3  
- Irrelevant claims：句子在字面上為真，但對環境偏好選擇沒有信息含量（典型例子是聲稱 “CFC-free” 而 CFC 已被法律禁止）。citeturn10view0turn11view0  
- Hidden tradeoffs：只強調單一綠色屬性，忽略生命週期/價值鏈其他重大衝擊；英國廣告規範與多份政策方向都強調「全生命週期」語境，否則容易誤導。citeturn10view0turn15view5  
- No proof：主張缺乏容易取得的支持資訊或可靠第三方驗證。citeturn10view0turn11view0turn15view2  
- Lesser of two evils：在本質高環境衝擊/高外部成本的品類中，把相對改良包裝成整體“綠”。citeturn10view0turn15view5  
- Fake labels：透過文字或圖像營造第三方背書的印象，但實際不存在該背書；監管文件對「認證/標章」的誤導性使用有明確警示。citeturn10view0turn11view0turn15view3  
- Outright lies：內容為虛假（例如杜撰數據、偽造認證）。citeturn10view0turn11view0  

在歐盟趨勢面，entity["organization","European Commission","eu executive body"] 曾引用研究指出：綠色宣稱中 53% 提供含糊、誤導或缺乏依據的信息，40% 沒有支持證據；同時「標章很多但驗證薄弱」也是政策推動的核心理由。citeturn4search3turn4search7  
此外，entity["organization","European Union","supranational union"] 的「Empowering consumers for the green transition」新規方向包含：禁止無法證明的含糊環境宣稱、禁止不可靠自願性標章。citeturn15view3turn4search1turn4search10

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["seven sins of greenwashing infographic","greenwashing vague claim eco friendly example","green claims code guidance infographic","ESG report sustainability claim example"],"num_per_query":1}

## 輸入資料到可運算資料的轉換管線

你要的「可直接丟給生成 code 模型」的程度，關鍵是把每一步的 I/O 格式固定。以下採用「元素級（layout element）→ 段落/句子 → 宣稱（claim）→ 證據（evidence）→ 判決（verdict）」的資料模型。

### 使用者輸入

輸入由兩類文件構成：

A. ESG report（通常 PDF）。  
B. 可選的參考/驗證資訊（可上傳 PDF/HTML/CSV、手動貼 URL、或手動填入聲明與證據）。

系統需要支援兩種 PDF 情況：可抽取文字的機器版 PDF、以及掃描版 PDF（需要 OCR）。工程上建議一律走「先做 layout-aware 解析」：保留頁碼、段落位置、標題階層、表格與註腳，後續才能把模型判決對回原文位置。

### PDF/文件解析建議技術與輸出

建議採「雙解析器」：  
主解析器用 `unstructured.partition.pdf` 把 PDF 切成元素（Title、NarrativeText、ListItem、Table…）。該工具的核心概念是把原始文件拆成帶類型的元素，讓後續決定保留哪些文本與保留哪些 metadata。citeturn8search0turn8search4turn8search2  
輔助解析器用 entity["organization","PyMuPDF","python pdf library"]（fitz）補足頁面文字順序與區塊資訊，並在需要時使用 `sort=True` 做閱讀順序排序。citeturn8search1turn8search3turn8search15  
表格抽取若要更穩定，可用 `camelot` 的 lattice/stream（取決於表格是否有線框）。citeturn8search9turn8search13

**解析輸出（Document Elements）格式固定成 JSONL：**

```json
{
  "doc_id": "uuid",
  "source_type": "esg_report|user_evidence",
  "file_name": "report.pdf",
  "page": 12,
  "element_id": "uuid",
  "element_type": "Title|NarrativeText|ListItem|Table|Footer|Header",
  "text": "....",
  "bbox": [x0, y0, x1, y1],
  "section_path": ["環境", "氣候變遷", "溫室氣體排放"],
  "table_html": null,
  "table_cells": null,
  "language": "zh|en|mixed",
  "hash": "sha256..."
}
```

此步的原則是：**任何後續可疑判決都必須能回指到 page + bbox 或至少 page + element_id**，才符合你要的互動式 UI（點類別 → 顯示原文與高亮）。

### Chunking 與建立檢索索引（RAG/證據檢索的底座）

把 element 再加工成兩種粒度：

- `paragraph_chunk`：以同一 section_path 下連續 NarrativeText/ListItem 合併，目標長度例如 300–800 中文字（或 200–500 tokens），保留原始 element_id 列表。  
- `sentence_unit`：對 paragraph 做句子切分，保留 sentence offset（start_char, end_char）。

同時建立三個索引：

- `vector_index`：用多語 embedding，供語意檢索。BGE-M3 的論文與模型卡都強調其多語與長文本能力，適合你這種中英混雜 ESG 報告；也可以替換成 multilingual E5。citeturn6search0turn6search2turn6search1turn6search5  
- `bm25_index`：字面檢索，對「標章名稱、法規名、認證編號、Scope 1/2/3」等專有詞特別有效。  
- `reranker`：對 top-k 的段落做二階段重排，建議用 BGE reranker v2 m3 或 SentenceTransformers CrossEncoder 類型。citeturn7search0turn7search9turn7search1  

RAG 的基本思想是「生成/推理」之前用檢索把相關片段取回，並把可追溯的片段當作依據。這個方向在知識密集任務的研究中被系統化描述。citeturn5search4turn5search0

## 核心偵測模組的分工設計

這裡把七類拆成可實作的模組，每個模組回傳同一種 `Verdict` 結構，然後由 `Aggregator` 統一寫入輸出。

### 全域前置模組：環境宣稱抽取與結構化

你後面所有七類，都依賴「先抽到宣稱」。建議做兩步：

第一步：ESG/環境相關段落偵測（topic gate）。如果報告全都是 ESG，這步可以簡化；若包含大量治理/財務/人資內容，這步可以大幅減少誤報。現成模型例子：FinBERT-ESG 可把句子分到 E/S/G/None（偏英文）；ClimateBERT 有 climate detector（偏英文且模型卡提醒更適合 paragraph）。citeturn14search3turn14search8turn14search12  
中文情境建議直接用多語 embedding + 关键词組合 gate，外加 local LLM 做少量校正。

第二步：宣稱句抽取（claim extraction）。輸出格式：

```json
{
  "claim_id": "uuid",
  "chunk_id": "uuid",
  "sentence_id": "uuid",
  "claim_text": "...",
  "claim_type": "environmental_benefit|net_zero_target|recycling|offset|energy_efficiency|other",
  "entities": [{"type":"CERT_LABEL","value":"..."}],
  "quantities": [{"value":30,"unit":"%","what":"recycled_content"}],
  "time": {"year":2025,"range":null}
}
```

在綠漂 NLP 綜述中，「先做 climate-related / claim about company / claim characteristics」這類子任務是常見可操作路徑。citeturn13view0

### 模組一：Vague claims 偵測（可離線訓練，低依賴外部資料）

**主要訊號：**  
（a）模糊形容詞密度高（eco-friendly、green、永續、友善…），（b）缺少方法（how）、範圍（where/which part）、基準（compared to what）、量化指標（how much）。監管指南對「泛稱環境利益」的風險與需限定性，有長篇具體規範。citeturn11view0turn15view5turn15view3

**建議實作：**  
- 規則特徵：模糊詞 lexicon、是否含數字與單位、是否含方法詞（盤查、LCA、第三方查核…）。  
- 小模型分類器：用 XLM-R 或中文 RoBERTa 做二元分類（vague vs specific）。  
- LLM 補充：讓 LLM 回答「這段主張的可驗證條件是什麼」，並產生 missing fields（範圍/基準/方法/數據來源）。

**輸出：** severity 分數一般由 (模糊詞數量, 缺失欄位數量) 組合決定。

### 模組二：No proof 偵測（多數可離線完成，但會產生“需要補證據”任務）

**主要訊號：** claim 周邊缺乏支持資訊或可信第三方驗證；TerraChoice 定義直接把「容易取得的支持資訊」與「可靠第三方認證」當判準。citeturn10view0

**建議實作：**  
- 內文證據檢索：以 claim 當 query，在 ESG report 的 vector_index/bm25_index 找 top-k 可能的「數據表、指標、審驗聲明、方法學」。  
- 證據評分：用 reranker 判定 evidence 是否真的支持 claim。citeturn7search9turn7search5  
- 結構化檢查：若 claim 涉及排放，檢查是否給出 Scope、邊界、年度、基準年、核算方法。citeturn15view0turn15view1  

**特別針對碳排與 Scope 的“proof checklist”：**  
entity["organization","ISSB","ifrs sustainability board"] 的 IFRS S2 明確要求 Scope 1/2/3 的揭露，且 Scope 的量測要依據 entity["organization","Greenhouse Gas Protocol","corporate ghg standard"]。citeturn15view0turn15view1  
因此實作上可以把「不揭露 Scope 分解」或「揭露但缺少邊界/方法」當作 no proof 的加權風險。

### 模組三：Hidden tradeoffs 偵測（需要跨段落對照，部分情境需要外部資料）

**主要訊號：**只強調單一綠屬性，忽略其他更重大衝擊。TerraChoice 對 hidden trade-off 的例子直接談到生命週期其他影響可能更重要。citeturn10view0  
entity["organization","Advertising Standards Authority","uk ads regulator"] 也明確要求環境宣稱要以全生命週期為基礎，否則容易讓消費者對整體環境影響產生誤解。citeturn15view5

**建議實作：**  
- 同章節對照：若段落宣稱「包裝可回收」，檢索同產品/同事業線是否揭露主要排放來源；若整份報告的排放主體是製程/能源/供應鏈，卻大量宣傳包裝或紙張，給出 hidden tradeoff risk。  
- “Materiality gap” 指標：用段落主題分佈（topic modeling 或 embedding clustering）找「宣稱密度高」但「數據/負面揭露低」的區域。綠漂 NLP 綜述提到 selective disclosure / cherry-picking、cheap talk 等概念可用於這種落差偵測。citeturn13view0  
- 類別知識（可選）：對高衝擊產業（航空、石化、水泥…）設定優先檢查清單，強制檢索 Scope 3、能源、原料等段落。

**輸出形態：**  
Hidden tradeoffs 很難只靠單一段落說清楚，建議輸出「段落風險 + 跨段落對照報告」，UI 仍能點到觸發段落，但旁邊附上“缺口對照”。

### 模組四：Irrelevant claims 偵測（需要「可更新知識庫」，API 情境可即時查）

**主要訊號：**主張在語意上為真但沒有信息含量，例如 TerraChoice 的 “CFC-free” 例子。citeturn10view0  
此類偵測要靠「禁用/普遍法規要求」的知識。

**建議實作：**  
- 建一個 `irrelevance_kb`（本地 JSON/SQLite）：包含「常見無效宣稱 → 條件、原因、參考來源」。例如：CFC、某些早已普及的合規項。  
- Claim 正規化：把 claim 抽取到 (object, attribute, qualifier) 後，對照 KB 規則。  
- API case：若 KB 沒命中，觸發即時查詢（例如查某物質是否已禁止、某標準是否強制），把查詢結果寫入 evidence。

### 模組五：Fake labels 偵測（強依賴外部驗證或離線白名單）

**主要訊號：**標章/認證給人第三方背書印象但實際不存在。TerraChoice 對「worshiping false labels」是明確定義；entity["organization","Federal Trade Commission","us consumer regulator"] 的 Green Guides 也明確提到：宣稱被獨立第三方背書/認證若不實，屬欺瞞；且使用不說明依據的環境認證/印章會讓人解讀成廣泛環境利益宣稱。citeturn10view0turn11view0  
歐盟新規方向同樣明確要禁止不可靠的自願性永續標章。citeturn15view3turn4search1

**建議實作：**  
- 認證實體抽取：NER + regex（如 “certified”, “認證”, “標章”, “seal”, “ISO”, “FSC”…）以及圖像標章（若你要做更高階，可從 PDF 抽圖再做 logo OCR/分類）。  
- 白名單知識庫：建 `label_registry`，包含常見標章的官方名稱、發證機構、驗證方式、查詢入口/規則。  
- 驗證策略：  
  - API case：對可疑標章做即時查詢（官方網站/公開 registry），把“查不到”當作強風險訊號。  
  - 無 API case：用離線白名單與使用者提供的補充資料。對白名單外的標章，輸出 “needs verification”。

補充：若你要把“可靠第三方標章”與“自稱標章”區隔，可以把 entity["organization","International Organization for Standardization","standards body"] 的 Type I ecolabel（ISO 14024）概念作為分類參考；例如 entity["organization","EU Ecolabel","eu ecolabel scheme"] 是 ISO 14024 Type I、多準則、第三方驗證，且明確採生命週期視角。citeturn15view4  

### 模組六：Lesser of two evils 偵測（多數可離線完成，需產業/品類辨識）

**主要訊號：**在整體高衝擊品類中強調相對改良；TerraChoice 給出 “organic cigarettes” 類例。citeturn10view0

**建議實作：**  
- 產業/品類分類器：從 ESG report 公司簡介、產品線、營收結構抽取 principal business。  
- 規則：對高風險品類（菸草、賭博、fast fashion、化石燃料等）設 “category risk”；當偵測到綠色宣稱就把 lesser-of-two-evils 類別打開。  
- 解釋模板：輸出要避免做道德判斷，直接指出「該品類存在高度環境外部成本，單點改良容易讓讀者誤判整體衝擊」。

### 模組七：Outright lies 偵測（最難，建議拆成兩種可落地子模組）

Outright lies 很難在 hackathon 時間內做成高置信度的“抓謊器”。比較可落地的兩個子模組是：

子模組 A：內部一致性/數據矛盾偵測（不需要外部資料）  
- 抽取同一指標（例如總排放、再生能源占比）在不同章節/表格的數值。  
- 若相同年度同一口徑出現矛盾，輸出 “internal inconsistency” 並指回兩個原文位置。  
- 這類輸出在 UI 很好呈現，且不需要你宣稱“外部事實為何”。

子模組 B：可驗證宣稱的外部核驗（強依賴即時查詢或使用者提供證據）  
- 對 “我們通過 X 認證”“我們達到 Y 等級”“我們碳中和” 等可核驗主張，建立 `checkworthy_claim`，然後：  
  - API case：即時查官方 registry / 可信資料源。  
  - 無 API case：只做「需要核驗」的任務清單，要求使用者補文件或鏈接。

這樣做符合綠漂 NLP 綜述指出的實務難點：綠色宣稱常跨多段落甚至跨多資料源，正確判斷的第一難題是把需要的元素蒐集齊。citeturn13view0  

## 兩種部署情境的完整系統架構

兩個 case 的差異建議只放在「推理器（Reasoner）與外部證據介面（Evidence Connectors）」，其餘資料管線共用，這樣你可以維持同一套 UI/輸出結構。

### Case A：可使用外部 LLM API 的混合式架構

目標：最高準確度、最好解釋品質、允許即時核驗。代價是費用與資料外送風險。

**核心元件（以資料流表示）：**

1) Ingestion Service  
- ESG report / evidence files 上傳 → `document_elements.jsonl`  
- 同步建立 `paragraph_chunks.jsonl`, `sentences.jsonl`

2) Indexing Service  
- 建 `vector_index`（多語 embedding）+ `bm25_index` + `reranker`  
- Reranker 用於 “claim → 找 supporting evidence” 與 “跨段落對照”

3) Claim Extractor  
- 規則 + 小模型 + LLM（用 function-calling/JSON schema 強制輸出欄位）  
- 把 claim 寫入 `claims.parquet`

4) Detector Orchestrator（七類模組）  
- 每個模組產生 `verdicts.parquet`，欄位統一（見下一節輸出 schema）  
- 對 fake labels / outright lies / irrelevance 觸發 Evidence Connectors

5) Evidence Connectors（即時核驗）  
- Certification lookup：查標章/認證官方 registry  
- Regulation lookup：查某主張是否屬普遍法規要求  
- Corporate facts：查公司是否真的有某認證/某承諾（此部分要謹慎選資料源）

6) LLM Reasoner（最後一哩）  
- 讀取：claim + top evidence + detectors 的中間結果  
- 輸出：  
  - 最終判決（類別、置信、嚴重度）  
  - 可讀解釋（限制只能引用 evidence 與原文）  
  - Missing-evidence checklist（給使用者補資料）

為了降低“生成型模型憑空補信息”的風險，Reasoner 應採取 RAG：每一段解釋都要綁定 evidence span；RAG 的目的之一就是提供決策依據與可追溯性。citeturn5search4turn5search0

### Case B：不可使用外部 API、只能使用 local LLM 的架構

目標：成本可控與資料不出域。代價是即時核驗能力下降，因此要把「需要核驗」輸出做得非常強。

**核心做法：**  
- 把 Case A 的 LLM API Reasoner 換成 local LLM（建議中文能力強、可長上下文），並把 Evidence Connectors 改成「離線 KB + 使用者上傳證據」兩條路。  
- 推理與 serving 用 vLLM 或 llama.cpp 類工具鏈部署常見 open-weight LLM；vLLM 主打高吞吐推理與 serving，llama.cpp 主打在多硬體上簡化本地推理。citeturn5search1turn5search2turn5search3

**離線知識庫設計（此處是 Case B 成敗關鍵）：**  
- `label_registry.sqlite`：常見標章與查核規則（可人工先建 100–300 個）  
- `irrelevance_kb.sqlite`：常見“無信息含量宣稱”的條目  
- `regulatory_principles.md`：把監管原則摘要成可被 local LLM 檢索的短文（例如“泛稱綠色宣稱需可佐證”“標章不得誤導”）。英國 entity["organization","Competition and Markets Authority","uk competition regulator"] 的 Green Claims Code 就是很典型的可引用原則集。citeturn15view2turn15view5turn11view0  

**Case B 的正確性策略：**  
- 對需要外部查核的類別（fake labels / outright lies / 低信息量的法規依賴）一律輸出 “verification_required”，並產出具體查核任務（查哪個 registry、要什麼文件、缺什麼欄位）。  
- 解釋文字限制在「原文 + 離線原則 + 使用者上傳證據」。  
- 系統對“事實真假”的措辭要更保守，偏向“可疑/需核驗”。

## 輸出資料結構與前端可互動呈現的連通關係

你要的 UI 是「七類面向」卡片 → 點進去看到原文與判決。這需要你在輸出中同時保留：

- 類別索引（category → items）  
- item → document span（page + bbox 或 element_id 列表）  
- 模型解釋 → evidence span（指回 report 的其他段落或外部證據）

建議最終輸出是一個 `final_report.json`（另存 parquet 便於查詢），概念如下：

```json
{
  "run_id": "uuid",
  "doc_id": "uuid",
  "lang": "zh-TW",
  "categories": [
    {
      "category": "vague_claims",
      "display_name": "Vague claims",
      "summary": "共標記 12 段，主要缺少量化指標與邊界定義。",
      "items": [
        {
          "item_id": "uuid",
          "severity": 0.78,
          "confidence": 0.74,
          "verdict": "flagged|pass|needs_verification",
          "span": {
            "page": 15,
            "element_ids": ["...","..."],
            "bbox_list": [[...],[...]],
            "quote": "原文摘錄..."
          },
          "claim": {
            "claim_id": "uuid",
            "text": "..."
          },
          "explanation": {
            "why": "以自然語言解釋風險點（限制引用 evidence）。",
            "missing_info": ["範圍界定", "量化指標", "基準年", "方法學/核證"]
          },
          "evidence": [
            {
              "evidence_type": "internal",
              "ref": {"page": 48, "element_id": "..."},
              "quote": "..."
            },
            {
              "evidence_type": "external",
              "ref": {"url": "...", "retrieved_at": "..."},
              "quote": "..."
            }
          ],
          "recommended_actions": [
            "要求公司提供量化指標與核算邊界",
            "要求提供第三方查核或可公開取得的支持資料"
          ]
        }
      ]
    }
  ],
  "global_findings": {
    "risk_heatmap": {
      "vague_claims": 0.61,
      "irrelevant_claims": 0.12,
      "hidden_tradeoffs": 0.44,
      "no_proof": 0.53,
      "lesser_two_evils": 0.07,
      "fake_labels": 0.09,
      "outright_lies": 0.05
    },
    "verification_tasks": [
      {
        "task_id": "uuid",
        "task_type": "certification_lookup|regulation_lookup|data_request",
        "prompt_to_user": "請上傳某認證證書或提供官方查詢鏈接",
        "linked_item_id": "uuid"
      }
    ]
  }
}
```

這份輸出可以直接映射到你的 UI：左側七類列表顯示 summary；點進類別後 list items；點 item 後用 span.bbox 高亮原文，右側顯示 explanation 與 evidence。

## 模型正確性優先的評估與校正方法

你明確說「以最佳表現為第一優先」。在 hackathon 期間可採取的最有效策略是：

第一，建立小而高品質的標註集。  
綠漂 NLP 綜述指出許多任務的標註一致性是核心問題，且沒有標準化正負樣本資料集會阻礙直接訓練與評估；因此你需要自己定義標準、並測 inter-annotator agreement（至少兩位標註者），把模糊案例列成“仲裁集”。citeturn13view0

第二，把評估拆成「claim 抽取」與「七類分類」與「證據支撐」三段。  
- Claim 抽取：以 recall 優先（漏掉宣稱會讓七類空掉）。  
- 七類分類：用 macro-F1 與 per-class precision/recall。  
- 證據支撐：做 human eval，檢查“模型解釋是否完全可由 evidence spans 支撐”。

第三，做置信度校正與「needs_verification」閥值。  
綠漂相關輸出永遠存在“不確定但需要查”的區域。對 fake labels / outright lies 等高風險類別，Case B 應更常輸出 needs_verification，以換取低誤報的“確定抓謊”。歐盟政策文本把「支持證據」「標章可靠性」當作核心問題，這也支持你把 verification_tasks 當一級輸出物。citeturn4search3turn15view3turn11view0