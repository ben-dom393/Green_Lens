# Green Lens — Pre-Built Knowledge Bases Specification

> All reference data is pre-built into the system. Users only upload ESG reports.
> This document specifies what data to collect and where to find it.

---

## Overview

| KB Name | File | Used by Modules | Priority | Est. entries |
|---------|------|-----------------|----------|-------------|
| Vague language lexicon | `data/vague_lexicon.json` | 1 (Vague Claims) | Phase 2 | ~200 terms |
| Regulatory principles | `data/regulatory_principles.json` | 2, 3, 5 | Phase 3 | ~50 principles |
| Irrelevance KB | `data/irrelevance_kb.json` | 3 (Irrelevant Claims) | Phase 5 | ~50 entries |
| Industry risk profiles | `data/industry_risk.json` | 4 (Lesser of Two Evils) | Phase 6 | ~15 sectors |
| Certification registry | `data/label_registry.json` | 6 (Fake Labels) | Phase 8 | 100-300 labels |
| Greenwashing examples | `data/greenwashing_cases.json` | All (few-shot, training, eval) | Phase 1 | 50-100 cases |
| Proof checklists | `data/proof_checklists.json` | 2 (No Proof) | Phase 3 | ~10 claim types |

---

## 1. Vague Language Lexicon (`vague_lexicon.json`)

### Format

```json
{
  "categories": {
    "weasel_quantifiers": {
      "description": "Vague quantity words that avoid specific numbers",
      "terms": ["some", "many", "most", "several", "numerous", "various", "certain", "a number of", "significant amount"]
    },
    "vague_commitments": {
      "description": "Commitment language that avoids binding promises",
      "terms": ["strive to", "aim to", "aspire to", "endeavor to", "seek to", "work towards", "committed to exploring", "plan to consider", "intend to evaluate"]
    },
    "peacock_terms": {
      "description": "Self-congratulatory language without evidence",
      "terms": ["industry-leading", "world-class", "best-in-class", "cutting-edge", "pioneering", "transformative", "revolutionary"]
    },
    "greenwashing_buzzwords": {
      "description": "Environmental terms that are meaningless without specific metrics",
      "terms": ["sustainable", "green", "eco-friendly", "clean", "responsible", "environmentally conscious", "nature-positive", "carbon-neutral", "net-zero"]
    },
    "modal_hedges": {
      "description": "Words expressing uncertainty or conditionality",
      "terms": ["could", "might", "may", "possibly", "potentially", "perhaps", "likely", "approximately"]
    },
    "vague_time": {
      "description": "Time references without specific dates",
      "terms": ["soon", "in the near future", "in due course", "over time", "going forward", "in the coming years", "by the end of the decade"]
    },
    "passive_attribution": {
      "description": "Attribution to unnamed sources or vague authority",
      "terms": ["it is believed", "it is said", "it is widely recognized", "studies show", "experts agree", "research indicates", "according to sources"]
    }
  }
}
```

### Where to build this from
- Academic hedge detection lexicons (CoNLL-2010 shared task word lists)
- Wikipedia "weasel words" guidelines
- FTC Green Guides list of problematic environmental terms
- UK CMA Green Claims Code examples of vague claims
- Team's own review of ESG reports

---

## 2. Regulatory Principles (`regulatory_principles.json`)

### Format

```json
{
  "principles": [
    {
      "id": "ftc_substantiation",
      "source": "FTC Green Guides (16 CFR Part 260)",
      "source_url": "https://www.ftc.gov/legal-library/browse/federal-register-notices/guides-use-environmental-marketing-claims-green-guides",
      "principle": "Environmental marketing claims must be substantiated by competent and reliable scientific evidence",
      "applies_to": ["vague_claims", "no_proof"],
      "detection_rule": "Any environmental benefit claim without referenced methodology, data, or third-party verification",
      "keywords": ["substantiation", "competent evidence", "reliable scientific evidence"]
    }
  ]
}
```

### Source documents to extract principles from

| Document | Download URL | Key principles |
|----------|-------------|----------------|
| FTC Green Guides (16 CFR 260) | [PDF](https://www.ftc.gov/sites/default/files/attachments/press-releases/ftc-issues-revised-green-guides/greenguides.pdf) or [eCFR](https://www.ecfr.gov/current/title-16/chapter-I/subchapter-B/part-260?toc=1) | Substantiation, specificity, qualification for each term |
| UK CMA Green Claims Code | [PDF](https://assets.publishing.service.gov.uk/media/61482fd4e90e070433f6c3ea/Guidance_for_businesses_on_making_environmental_claims_.pdf) or [GOV.UK](https://www.gov.uk/government/publications/green-claims-code-making-environmental-claims) | 6 principles: truthful, accurate, clear, not omit info, fair comparisons, substantiated |
| EU ECGT Directive 2024/825 | [EUR-Lex](https://eur-lex.europa.eu/eli/dir/2024/825/oj/eng) | Bans generic claims (eco, green, climate-friendly) without evidence; bans offset-only carbon neutral claims; bans unreliable sustainability labels. In force Sept 2026. |
| ISSB/IFRS S2 | [Standard PDF](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-a/issb-2023-a-ifrs-s2-climate-related-disclosures.pdf?bypass=on) | Scope 1/2/3 disclosure, methodology, boundaries, base year |
| GHG Protocol Corporate Standard | [PDF (3.5MB)](https://ghgprotocol.org/sites/default/files/standards/ghg-protocol-revised.pdf) | Scope definitions, calculation approaches, reporting requirements |
| GHG Protocol Scope 3 Standard | [PDF](https://ghgprotocol.org/sites/default/files/standards/Corporate-Value-Chain-Accounting-Reporing-Standard_041613_2.pdf) | Value chain emissions categories |
| UK ASA Environmental Claims Guidance | [ASA](https://www.asa.org.uk/advice-online/environmental-claims-general.html) | Marketing claims must not mislead, must be qualified |
| ISO 14021 (Type II ecolabels) | iso.org (paywalled) | Self-declared environmental claims must be verifiable |
| TCFD Recommendations | [FSB-TCFD](https://www.fsb-tcfd.org/) | Governance, strategy, risk management, metrics/targets |

### How to build
1. Download or access each source document
2. Extract key principles as structured entries
3. Map each principle to which detection modules it informs
4. Include specific keywords/patterns that trigger the principle

---

## 3. Irrelevance KB (`irrelevance_kb.json`)

### Format

```json
{
  "entries": [
    {
      "pattern": "CFC-free",
      "pattern_regex": "(?i)\\b(cfc[- ]?free|chlorofluorocarbon[- ]?free)\\b",
      "reason": "CFCs have been banned globally since 1987 under the Montreal Protocol. Claiming CFC-free is stating compliance with a universal legal requirement.",
      "regulation": "Montreal Protocol (1987)",
      "applies_to": "all products and industries",
      "sin_type": "irrelevant_claims"
    }
  ]
}
```

### Known irrelevant claim patterns to include

| Claim pattern | Why irrelevant | Regulation |
|--------------|----------------|------------|
| CFC-free | Banned globally since 1987 | Montreal Protocol |
| Lead-free paint | Banned in most countries | Various national laws |
| Asbestos-free | Banned or severely restricted | Various national laws |
| Mercury-free | Severely restricted | Minamata Convention (2013) |
| BPA-free (in some products) | Required by law in many jurisdictions | FDA, EU regulations |
| "Complies with environmental law" | All products must comply; this is not a distinction | General legal compliance |
| "We recycle our office paper" | Universal practice, negligible impact for large corporations | N/A |
| Ozone-friendly | ODS banned under Montreal Protocol | Montreal Protocol |

### Where to find more
- TerraChoice original studies (specific examples of irrelevant claims)
- EU Commission sweep studies
- UNEP lists of banned substances
- Stockholm Convention on Persistent Organic Pollutants

---

## 4. Industry Risk Profiles (`industry_risk.json`)

### Format

```json
{
  "sectors": [
    {
      "sector_id": "fossil_fuels",
      "display_name": "Fossil Fuels (Oil, Gas, Coal)",
      "keywords": ["oil", "gas", "petroleum", "coal", "refining", "crude", "natural gas", "LNG", "upstream", "downstream"],
      "sic_codes": ["1311", "1381", "2911", "4911"],
      "risk_level": "very_high",
      "primary_impacts": [
        "Scope 1 direct emissions from operations",
        "Scope 3 downstream combustion by end users",
        "Methane leakage",
        "Ecosystem disruption from extraction"
      ],
      "lesser_evil_note": "Single-point environmental improvements (e.g., reduced flaring, cleaner refining) do not offset the fundamental climate impact of the core product being sold.",
      "expected_material_topics": ["GHG emissions (all scopes)", "methane emissions", "transition risk", "stranded assets", "just transition"]
    }
  ]
}
```

### Sectors to include

| Sector | Risk level | Key environmental impact |
|--------|-----------|------------------------|
| Fossil fuels | Very high | GHG emissions (core product) |
| Tobacco | Very high | Health + deforestation + chemicals |
| Mining | High | Land degradation, water pollution, biodiversity |
| Cement / concrete | High | ~8% of global CO2 emissions |
| Aviation | High | High-altitude emissions, contrails |
| Fast fashion | High | Water use, chemical pollution, waste |
| Agrochemicals / pesticides | High | Biodiversity loss, water contamination |
| Steel / metals | High | Energy-intensive, Scope 1 emissions |
| Shipping / maritime | High | Fuel oil emissions, ballast water |
| Automotive (ICE) | Medium-high | Tailpipe emissions, manufacturing |
| Plastics / packaging | Medium-high | Petroleum-based, ocean pollution |
| Construction | Medium | Embodied carbon, waste |
| Banking / finance | Medium | Financed emissions (Scope 3 Cat 15) |
| Food & beverage | Medium | Agriculture emissions, packaging, supply chain |
| Technology | Medium-low | Energy use (data centers), e-waste, supply chain |

### Sources
- SASB Materiality Map (by industry)
- ISSB sector-specific requirements
- Carbon Disclosure Project (CDP) sector benchmarks
- Science Based Targets initiative (SBTi) sector guidance

---

## 5. Certification Registry (`label_registry.json`)

### Format

```json
{
  "certifications": [
    {
      "name": "ISO 14001",
      "official_org": "International Organization for Standardization",
      "type": "Management System Standard",
      "iso_ecolabel_type": null,
      "description": "Environmental management system certification. Certifies that an organization has an EMS in place, NOT that its environmental performance is good.",
      "common_misuse": "Claiming ISO 14001 implies low environmental impact — it only means the company has management processes in place.",
      "verification": "Third-party audit by accredited certification body",
      "lookup_url": "Check with national accreditation body",
      "valid_claim_example": "Our operations are ISO 14001:2015 certified by [accredited body]",
      "suspicious_claim_example": "We are ISO certified, proving our commitment to the environment"
    }
  ]
}
```

### Priority certifications to include (first 50)

**Environmental management:** ISO 14001, EMAS, ISO 50001 (energy)
**Carbon/climate:** SBTi validated targets, Carbon Trust Standard, PAS 2060 (carbon neutrality), Gold Standard, Verified Carbon Standard (VCS/Verra)
**Forestry/paper:** FSC, PEFC, SFI
**Buildings:** LEED, BREEAM, WELL, ENERGY STAR (buildings)
**Products:** EU Ecolabel, Blue Angel, Nordic Swan, ENERGY STAR (products), Cradle to Cradle, EPEAT
**Agriculture/food:** Organic (USDA, EU), Rainforest Alliance, Fair Trade, MSC (seafood), ASC (aquaculture)
**Textiles:** OEKO-TEX, GOTS, bluesign
**General:** B Corp, EcoVadis rating
**Chemical/safety:** REACH compliance, RoHS
**Circular economy:** How2Recycle, OK Compost

### Where to find data
- Ecolabel Index (ecolabelindex.com) — directory of 450+ ecolabels
- ISO catalog (iso.org)
- EU Ecolabel product groups (ec.europa.eu)
- Global Ecolabelling Network (globalecolabelling.net)

---

## 6. Greenwashing Cases (`greenwashing_cases.json`)

### Format

```json
{
  "cases": [
    {
      "case_id": "terraChoice_vague_001",
      "source": "TerraChoice Seven Sins Study",
      "sin_type": "vague_claims",
      "claim_text": "This product is all-natural and eco-friendly.",
      "context": "Consumer product packaging with no supporting details",
      "why_greenwashing": "Terms 'all-natural' and 'eco-friendly' are too broad to be meaningful. Arsenic and mercury are natural but toxic. No metrics, scope, or methodology provided.",
      "what_would_fix_it": "Specify which aspects are eco-friendly, with measurable criteria (e.g., '30% post-consumer recycled content, verified by [third party]')",
      "use_as": ["few_shot_example", "setfit_training", "evaluation"]
    }
  ]
}
```

### Sources for collecting cases

| Source | Type | How to access | Est. usable cases |
|--------|------|--------------|-------------------|
| ClimateBERT environmental_claims | 2,647 annotated sentences from corporate reports | [HuggingFace](https://huggingface.co/datasets/climatebert/environmental_claims) (MIT) | 2,647 sentences |
| ClimateBERT climate_specificity | Vague vs specific climate claims | [HuggingFace](https://huggingface.co/datasets/climatebert/climate_specificity) | Training data for Module 1 |
| A3CG (ACL 2025) | Aspect-action: implemented / planning / indeterminate | [GitHub](https://github.com/keanepotato/a3cg_greenwash) | Greenwashing classification |
| DizzyPanda1 GreenwashingDetectionDataset | Company, claim, accusation, certificates | [GitHub](https://github.com/DizzyPanda1/GreenwashingDetectionDataset) | Real cases with labels |
| CLIMATE-FEVER | 1,535 claims + 7,675 evidence pairs (supports/refutes/not enough) | [HuggingFace](https://huggingface.co/datasets/tdiggelm/climate_fever) | Fact-checking training |
| TerraChoice "Sins of Greenwashing" (2007-2010) | Original framework + product examples | [2010 PDF](https://www.twosides.info/wp-content/uploads/2018/05/Terrachoice_The_Sins_of_Greenwashing_-_Home_and_Family_Edition_2010.pdf), [2007 PDF](https://sustainability.usask.ca/documents/Six_Sins_of_Greenwashing_nov2007.pdf) | 20-30 |
| UK ASA rulings (environmental) | Structured rulings with claims + decisions | [ASA Rulings](https://www.asa.org.uk/codes-and-rulings/rulings.html) (search "environmental") | 30-50 |
| FTC enforcement cases | ~100 cases since 1991 with claim language + outcomes | [FTC page](https://www.ftc.gov/news-events/topics/truth-advertising/green-guides) | 30-50 |
| EU Commission sweep (2021) | 344 sustainability claims from 27 countries | [EU Sweeps](https://commission.europa.eu/live-work-travel-eu/consumer-rights-and-complaints/enforcement-consumer-protection/sweeps_en) | 10-20 |
| ClimaBench | Multi-task climate NLP benchmark | [HuggingFace](https://huggingface.co/datasets/iceberg-nlp/climabench) | Evaluation |
| Climate Policy Radar docs | Climate policy document corpus | [HuggingFace](https://huggingface.co/datasets/ClimatePolicyRadar/all-document-text-data) | Reference KB |
| Mendeley ESG & Greenwashing Dataset | ESG data with greenwashing annotations | [Mendeley](https://data.mendeley.com/datasets/vv5695ywmn/1) | Supplementary |

### Target: 8-16 labeled examples per greenwashing category for SetFit training
- 8-16 vague claims examples
- 8-16 no proof examples
- 8-16 irrelevant claims examples
- 8-16 lesser of two evils examples
- 8-16 hidden tradeoffs examples
- 8-16 fake labels examples
- 8-16 needs verification examples
- 8-16 NEGATIVE examples (legitimate, well-substantiated claims)

---

## 7. Proof Checklists (`proof_checklists.json`)

### Format

```json
{
  "claim_types": [
    {
      "type": "emissions_reduction",
      "description": "Claims about reducing GHG emissions",
      "required_evidence": [
        {"field": "scope", "description": "Which Scope (1, 2, 3 or combination)?"},
        {"field": "boundary", "description": "Organizational/operational boundary of the claim"},
        {"field": "base_year", "description": "What year is the reduction measured against?"},
        {"field": "methodology", "description": "Calculation methodology (GHG Protocol, ISO 14064, etc.)"},
        {"field": "verification", "description": "Third-party assurance/verification statement"},
        {"field": "absolute_vs_intensity", "description": "Is the reduction absolute or intensity-based?"},
        {"field": "exclusions", "description": "What is excluded from the boundary?"}
      ],
      "source": "ISSB/IFRS S2, GHG Protocol Corporate Standard"
    }
  ]
}
```

### Claim types to include

| Claim type | Key required evidence | Source standard |
|-----------|----------------------|----------------|
| Emissions reduction | Scope, boundary, base year, methodology, verification | ISSB S2, GHG Protocol |
| Net-zero / carbon neutral | Target year, interim targets, scope, offset methodology, residual emissions plan | SBTi Net-Zero Standard |
| Renewable energy | Percentage, source, RECs vs. on-site vs. PPA, additionality | GHG Protocol Scope 2 Guidance |
| Recycled content | Percentage, pre/post-consumer, certification, methodology | ISO 14021, FTC Green Guides |
| Water reduction | Baseline, boundary, measurement methodology, region context | CDP Water, GRI 303 |
| Biodiversity | Baseline assessment, metrics, location, methodology, monitoring | TNFD, GRI 304 |
| Circular economy | Definition used, metrics, boundary, waste hierarchy compliance | EU Circular Economy Action Plan |
| Supply chain sustainability | Scope (tier 1/2/3+), audit methodology, coverage percentage | Various |
| Waste reduction / zero waste | Definition, boundary, diversion rate methodology, landfill vs. incineration | Zero Waste International Alliance |
| Sustainable packaging | Material sourcing, recyclability testing, end-of-life infrastructure | FTC Green Guides, EU PPWR |
