# External Data Directory Structure

Place downloaded files in the appropriate folder below.

## Folder Structure

```
external/
├── regulatory/              ← Laws, standards, guidelines
│   ├── us/                  ← US regulations
│   │   └── FTC_Green_Guides.pdf          (download manually - blocked by anti-bot)
│   ├── uk/                  ← UK regulations
│   │   └── UK_CMA_Green_Claims_Code.pdf  ✓ downloaded
│   ├── eu/                  ← EU regulations
│   │   └── EU_Directive_2024_825.pdf     (download from EUR-Lex)
│   │   └── CSRD_ESRS_E1.pdf             (download from EFRAG)
│   └── international/       ← International standards
│       └── GHG_Protocol_Corporate_Standard.pdf  ✓ downloaded
│       └── GHG_Protocol_Scope3.pdf              (via download_datasets.py)
│       └── IFRS_S2_Climate_Disclosures.pdf      (via download_datasets.py)
│       └── SBTi_Net_Zero_Standard.pdf           (download from sciencebasedtargets.org)
│
├── cases/                   ← Known greenwashing cases & examples
│   ├── terrachoice/         ← TerraChoice Seven Sins studies
│   │   └── TerraChoice_Seven_Sins_2010.pdf  ✓ downloaded
│   │   └── TerraChoice_Six_Sins_2007.pdf    ✓ downloaded
│   ├── asa_rulings/         ← UK ASA advertising rulings (environmental)
│   │   └── (download from asa.org.uk/rulings - search "environmental")
│   ├── ftc_cases/           ← US FTC enforcement actions
│   │   └── (download from ftc.gov/green-guides)
│   ├── eu_sweep/            ← EU Commission sweep study results
│   │   └── (download from ec.europa.eu sweeps page)
│   └── news_examples/       ← News articles about greenwashing scandals
│       └── (save articles as .txt or .pdf)
│
├── datasets/                ← NLP datasets for training & evaluation
│   ├── huggingface/         ← Downloaded via download_datasets.py
│   │   └── climatebert__environmental_claims/
│   │   └── climatebert__climate_specificity/
│   │   └── climatebert__climate_detection/
│   │   └── climatebert__climate_sentiment/
│   │   └── climatebert__climate_commitments_actions/
│   │   └── tdiggelm__climate_fever/
│   └── github/              ← Cloned from GitHub
│       └── GreenwashingDetectionDataset/   ✓ cloned (97 real cases)
│       └── a3cg_greenwash/                 ✓ cloned (ACL 2025)
│
└── esg_reports/             ← Sample ESG reports for testing
    └── sample_reports/      ← Put company ESG report PDFs here
```

## What to download manually

### High priority (put in regulatory/)
- FTC Green Guides: https://www.ecfr.gov/current/title-16/chapter-I/subchapter-B/part-260
- EU Directive 2024/825: https://eur-lex.europa.eu/eli/dir/2024/825/oj/eng
- SBTi Net-Zero Standard: https://sciencebasedtargets.org/resources/files/Net-Zero-Standard.pdf
- SASB Materiality Map: https://sasb.ifrs.org/standards/materiality-map/

### Medium priority (put in cases/)
- ASA rulings: https://www.asa.org.uk/codes-and-rulings/rulings.html (search "environmental")
- FTC cases: https://www.ftc.gov/news-events/topics/truth-advertising/green-guides

### Nice to have (put in datasets/)
- Mendeley ESG dataset: https://data.mendeley.com/datasets/vv5695ywmn/1
- DAX ESG Media: https://www.kaggle.com/datasets/equintel/dax-esg-media-dataset
- Climate Policy Radar: https://huggingface.co/datasets/ClimatePolicyRadar/all-document-text-data

### Test reports (put in esg_reports/sample_reports/)
- Any S&P 500 company ESG/sustainability report PDFs
