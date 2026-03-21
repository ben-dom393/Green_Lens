"""
Download ESG/Sustainability Reports from S&P 500 Companies.

Downloads publicly available ESG reports from official corporate websites
into the Green_Lens/ESG report folder for greenwashing analysis testing.
"""

import os
import time
import requests
from pathlib import Path
from urllib.parse import unquote

# Target directory
OUTPUT_DIR = Path(r"C:\Users\owen3\OneDrive\Desktop\Warwick CS\Spark the Globe\Green_Lens\ESG report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common headers to avoid being blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}

# (filename, url) pairs - all from official corporate sources
REPORTS = [
    # ── Technology ──
    ("Microsoft_2024_Environmental_Sustainability_Report.pdf",
     "https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/Microsoft-2024-Environmental-Sustainability-Report.pdf"),

    ("Amazon_2024_Sustainability_Report.pdf",
     "https://sustainability.aboutamazon.com/2024-amazon-sustainability-report.pdf"),

    ("Intel_2024-25_Corporate_Responsibility_Report.pdf",
     "https://csrreportbuilder.intel.com/pdfbuilder/pdfs/CSR-2024-25-Full-Report.pdf"),

    ("Cisco_FY2024_Purpose_Report.pdf",
     "https://www.cisco.com/c/dam/m/en_us/about/csr/esg-hub/_pdf/purpose-report-2024.pdf"),

    ("Salesforce_FY2024_Stakeholder_Impact_Report.pdf",
     "https://a.sfdcstatic.com/assets/prod/documents/white-papers/salesforce-fy24-stakeholder-impact-report.pdf"),

    ("Oracle_2024_Environmental_Social_Impact_Report.pdf",
     "https://www.oracle.com/a/ocom/docs/environmental-and-social-impact-report-2024.pdf"),

    ("Accenture_2024_CDP_Climate_Response.pdf",
     "https://www.accenture.com/content/dam/accenture/final/corporate/company-information/document/Accenture-CDP-2024.pdf"),

    ("Applied_Materials_2024_Impact_Report.pdf",
     "https://www.appliedmaterials.com/content/dam/site/company/csr/doc/2024_impact_report.pdf.coredownload.inline.pdf"),

    # ── Automotive & EV ──
    ("Tesla_2024_Impact_Report_Highlights.pdf",
     "https://www.tesla.com/ns_videos/2024-tesla-impact-report-highlights.pdf"),

    # ── Retail & Consumer ──
    ("Walmart_FY2025_ESG_Report.pdf",
     "https://corporate.walmart.com/content/dam/corporate/documents/esgreport/2025/FY2025-Walmart-ESG-Report.pdf"),

    ("Target_2024_Sustainability_Governance_Report.pdf",
     "https://corporate.target.com/getmedia/e2d80340-eb9f-43a7-a84c-219280aa5ba4/2024-Sustainability-and-Governance-Report.pdf"),

    ("Home_Depot_2024_ESG_Report.pdf",
     "https://corporate.homedepot.com/sites/default/files/2024-08/2024_ESG_Report_The_Home_Depot.pdf"),

    ("Costco_2025_Sustainability_Report.pdf",
     "https://s201.q4cdn.com/287523651/files/doc_financials/2025/ar/costco-sustainability-report-2025.pdf"),

    ("Lowes_2024_Corporate_Responsibility_Report.pdf",
     "https://corporate.lowes.com/sites/lowes-corp/files/2025-07/2024%20Reports/Lowes_2024_CRR_6.30.25.pdf"),

    ("Nike_FY2024_Sustainability_Data.pdf",
     "https://media.about.nike.com/files/460d3198-9609-4f5b-b18c-d4e7aa570bff/FY24_NIKE,-Inc.-Sustainability-Data.pdf"),

    ("Nike_FY2023_Impact_Report.pdf",
     "https://media.about.nike.com/files/2fd5f76d-50a2-4906-b30b-3b6046f36ebf/FY23_Nike_Impact_Report.pdf"),

    ("Starbucks_Fiscal_2024_Global_Impact_Report.pdf",
     "https://about.starbucks.com/uploads/2025/05/Starbucks-Fiscal-2024-Global-Impact-Report.pdf"),

    # ── Food & Beverage ──
    ("McDonalds_2024-2025_Purpose_Impact_Report.pdf",
     "https://corporate.mcdonalds.com/content/dam/sites/corp/nfl/pdf/McDonalds_PurposeImpact_ProgressReport_2024_2025.pdf"),

    ("McDonalds_2023-2024_Climate_Resiliency_Summary.pdf",
     "https://corporate.mcdonalds.com/content/dam/sites/corp/nfl/pdf/McDonalds-2023-Climate-Resiliency-Summary.pdf"),

    ("Coca_Cola_2024_Environmental_Update.pdf",
     "https://www.coca-colacompany.com/content/dam/company/us/en/reports/2024-environmental-update/2024-environmental-update.pdf"),

    ("PepsiCo_2024_Green_Bond_Report.pdf",
     "https://www.pepsico.com/docs/default-source/sustainability-and-esg-topics/pepsico-2024-green-bond-report.pdf"),

    ("General_Mills_2024_Global_Responsibility_Report.pdf",
     "https://s29.q4cdn.com/993087495/files/doc_downloads/global_responsibility_report/2024/General_Mills-Global_Responsibility_2024-1.pdf"),

    ("Kraft_Heinz_2024_ESG_Report.pdf",
     "https://www.kraftheinzcompany.com/esg/pdf/KraftHeinz-2024-ESG-Report.pdf"),

    # ── Financial Services ──
    ("JPMorgan_Chase_2024_Sustainability_Report.pdf",
     "https://www.jpmorganchase.com/content/dam/jpmorganchase/documents/about/jpmc-sustainability-report-2024.pdf"),

    ("Bank_of_America_2024_Sustainability_Report.pdf",
     "https://about.bankofamerica.com/content/dam/about/report-center/esg/2024/Sustainability_at_Bank_of_America_2024_Report.pdf"),

    ("Morgan_Stanley_2024_Sustainability_Report.pdf",
     "https://www.morganstanley.com/content/dam/msdotcom/en/assets/pdfs/Morgan_Stanley_2024_Sustainability_Report.pdf"),

    ("Citigroup_2024_Global_Sustainability_Report.pdf",
     "https://www.citigroup.com/rcs/citigpa/storage/public/global-sustainability-report-2024.pdf"),

    ("Wells_Fargo_2024_Sustainability_Governance_Report.pdf",
     "https://www.banktrack.org/download/sustainability_governance_report_1/240808_wells_fargo_sustainabilityandgovernancereport.pdf"),

    ("Goldman_Sachs_2023_Sustainability_Report.pdf",
     "https://www.goldmansachs.com/our-commitments/sustainability/2023-sustainability-report/multimedia/report.pdf"),

    ("Mastercard_2024_Impact_Report.pdf",
     "https://www.mastercard.com/content/dam/mccom/shared/for-the-world/corporate-impact/pdfs/mastercard-2024-impact-report.pdf"),

    ("Visa_2023_Corporate_Responsibility_Report.pdf",
     "https://corporate.visa.com/content/dam/VCOM/regional/na/us/about-visa/documents/2023-corporate-responsibility-sustainability-report.pdf"),

    # ── Healthcare & Pharma ──
    ("Johnson_Johnson_2024_ESG_Summary.pdf",
     "https://healthforhumanityreport.jnj.com/2024/_assets/downloads/johnson-johnson-2024-esg-summary.pdf"),

    ("UnitedHealth_2024_Sustainability_Executive_Summary.pdf",
     "https://www.unitedhealthgroup.com/content/dam/sustainability-report/2024/pdf/2024-SR_Executive-Summary.pdf"),

    ("Pfizer_2023_ESG_Performance.pdf",
     "https://cdn.pfizer.com/pfizercom/Pfizer_2023_ESG_Performance_13MAR2024.pdf"),

    ("Johnson_Controls_2024_Sustainability_Report.pdf",
     "https://www.johnsoncontrols.com/-/media/project/jci-global/johnson-controls/us-region/united-states-johnson-controls/corporate-sustainability/reporting-and-policies/documents/2024-sustainability-report.pdf"),

    # ── Energy & Utilities ──
    ("ExxonMobil_2024_Sustainability_Report.pdf",
     "https://corporate.exxonmobil.com/-/media/global/files/sustainability-report/2024/sustainability-report.pdf"),

    ("ExxonMobil_2024_Advancing_Climate_Solutions.pdf",
     "https://corporate.exxonmobil.com/-/media/global/files/advancing-climate-solutions/2024/2024-advancing-climate-solutions-report.pdf"),

    ("ConocoPhillips_2024_Sustainability_Report.pdf",
     "https://static.conocophillips.com/files/resources/2024-sustainability-report.pdf"),

    ("Chevron_2024_Sustainability_Highlights.pdf",
     "https://www.chevron.com/-/media/shared-media/documents/chevron-sustainability-highlights-2024.pdf"),

    ("NextEra_Energy_2024_Sustainability_Report.pdf",
     "https://www.investor.nexteraenergy.com/~/media/Files/N/NEE-IR/Sustainability/2024%20Sustainability/NextEra%20Energy%20Sustainability%20Report%202024.pdf"),

    ("Duke_Energy_2024_Impact_Report.pdf",
     "https://s201.q4cdn.com/583395453/files/doc_downloads/2025/10/impact-report-2024-final.pdf"),

    ("Dominion_Energy_2024_Sustainability_Report.pdf",
     "https://sustainability.dominionenergy.com/SCR-Report-2024.pdf"),

    # ── Aerospace & Defense ──
    ("Boeing_2024_Sustainability_Social_Impact_Report.pdf",
     "https://www.boeing.com/content/dam/boeing/boeingdotcom/sustainability/pdf/2024-boeing-sustainability-socialImpact-report.pdf"),

    ("Lockheed_Martin_2024_Sustainability_Report.pdf",
     "https://sustainability.lockheedmartin.com/sustainability/content/2024-Sustainability-Performance-Report.pdf"),

    ("Northrop_Grumman_2024_Sustainability_Report.pdf",
     "https://cdn.northropgrumman.com/-/media/Project/Northrop-Grumman/ngc/who-we-are/Sustainability/2024-Sustainability-Report.pdf"),

    # ── Transportation & Logistics ──
    ("FedEx_2024_ESG_Report.pdf",
     "https://www.fedex.com/content/dam/fedex/us-united-states/sustainability/gcrs/FedEx_2024_ESG_Report.pdf"),

    ("American_Airlines_2024_Sustainability_Report.pdf",
     "https://s202.q4cdn.com/986123435/files/images/esg/American-Airlines-Sustainability-Report-2024.pdf"),

    ("UPS_2024_GRI_Sustainability_Report.pdf",
     "https://about.ups.com/content/dam/upsstories/images/our-impact/reporting/2024-UPS-GRI-Report.pdf"),

    # ── Industrials ──
    ("Honeywell_2024_Impact_Report.pdf",
     "https://www.honeywell.com/content/dam/honeywellbt/en/documents/downloads/hon-2024-impact-report.pdf"),

    ("John_Deere_2024_Business_Impact_Report.pdf",
     "https://www.deere.com/assets/pdfs/common/our-company/sustainability/business-impact-report-2024.pdf"),

    ("John_Deere_2024_Sustainability_Data.pdf",
     "https://www.deere.com/assets/pdfs/common/our-company/sustainability/data-book-2024.pdf"),

    # ── Media & Entertainment ──
    ("Disney_2024_Sustainability_Social_Impact_Report.pdf",
     "https://thewaltdisneycompany.com/app/uploads/2025/05/2024-SSI-Report.pdf"),

    # ── Consumer Goods ──
    ("Procter_Gamble_2023_Citizenship_Report.pdf",
     "https://downloads.ctfassets.net/oggad6svuzkv/7PrU3Jq8gGsPAUl7fgZiR/919bd0309ccc76fd6365e047ba58ee5a/P_G_2023_Citizenship_Report_PDF.pdf"),

    # ── Financial Data & Services ──
    ("SP_Global_2023_Impact_Report.pdf",
     "https://www.spglobal.com/content/dam/spglobal/corporate/en/documents/organization/who-we-are/sp-global-impact-report-2023.pdf"),

    # ── Additional Bank Reports ──
    ("Bank_of_America_2025_Sustainability_Report.pdf",
     "https://about.bankofamerica.com/content/dam/about/report-center/esg/2025/SustainabilityatBofA2025_WCAG2.2_121625.pdf"),

    ("JPMorgan_Chase_2024_Climate_Report.pdf",
     "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/documents/Climate-Report-2024.pdf"),

    # ── Telecom ──
    ("Verizon_2024_Green_Bond_Impact_Report.pdf",
     "https://www.verizon.com/about/sites/default/files/Verizon-green-bond-impact-report-february-2024.pdf"),

    # ── Additional Energy ──
    ("Chevron_2023_Sustainability_Report.pdf",
     "https://www.chevron.com/-/media/shared-media/documents/chevron-sustainability-report-2023.pdf"),

    # ── responsibilityreports.com hosted PDFs ──
    ("Caseys_FY2024_Sustainability_Report.pdf",
     "https://www.responsibilityreports.com/HostedData/ResponsibilityReports/PDF/NASDAQ_CASY_2024.pdf"),

    ("Freeport_McMoRan_2024_Sustainability_Report.pdf",
     "https://www.responsibilityreports.com/HostedData/ResponsibilityReports/PDF/NYSE_FCX_2024.pdf"),

    ("Canon_2024_Sustainability_Report.pdf",
     "https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/c/NYSE_CAJ_2024.pdf"),

    ("PayPal_2023_Global_Impact_Report.pdf",
     "https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/p/NASDAQ_PYPL_2023.pdf"),

    # ── More from web search results ──
    ("Honeywell_ESG_Report.pdf",
     "https://www.honeywell.com/content/dam/honeywellbt/en/documents/downloads/hon-esg-report.pdf"),

    ("AbbVie_ESG_Disclosure_Supplement.pdf",
     "https://www.abbvie.com/content/dam/abbvie-com2/pdfs/esg-disclosure-supplement.pdf"),

    ("Walmart_FY2025_Sustainable_Commodities_Report.pdf",
     "https://corporate.walmart.com/content/dam/corporate/documents/esgreport/2025/FY2025-Walmart-Sustainable-Commodities-Report.pdf"),

    ("Bristol_Myers_Squibb_2021_ESG_Report.pdf",
     "https://www.bms.com/assets/bms/us/en-us/pdf/bmy-2021-esg-report.pdf"),

    ("McDonalds_2023-2024_Progress_Summary.pdf",
     "https://corporate.mcdonalds.com/content/dam/sites/corp/nfl/pdf/McDonald%E2%80%99s%20Progress%20Summary%202023-2024.pdf"),

    ("Intel_2023-24_CSR_Executive_Summary.pdf",
     "https://csrreportbuilder.intel.com/pdfbuilder/pdfs/CSR-2023-24-Executive-Summary.pdf"),

    # ── More responsibilityreports.com ──
    ("JBGS_Smith_2024_Sustainability_Report.pdf",
     "https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/j/NYSE_JBGS_2024.pdf"),

    ("TransUnion_2023_Sustainability_Report.pdf",
     "https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/t/NYSE_TRU_2023.pdf"),

    # ── Additional well-known companies ──
    ("Lowes_2023_Corporate_Responsibility_Report.pdf",
     "https://corporate.lowes.com/sites/lowes-corp/files/2024-09/Lowes_2023_CRR_9.30.24.pdf"),

    ("Wells_Fargo_2024_Climate_Report.pdf",
     "https://www.banktrack.org/download/climate_report_2024_2/240809_wells_fargo_climatedisclosure.pdf"),

    ("Cisco_2024_SASB_Response.pdf",
     "https://www.cisco.com/c/dam/m/en_us/about/csr/esg-hub/_pdf/2024-sasb.pdf"),

    ("ConocoPhillips_2023_Sustainability_Report.pdf",
     "https://static.conocophillips.com/files/resources/conocophillips-2023-sustainability-report.pdf"),

    ("Northrop_Grumman_2024_Sustainability_Executive_Summary.pdf",
     "https://cdn.northropgrumman.com/-/media/Project/Northrop-Grumman/ngc/who-we-are/Sustainability/2024-Sustainability-Report-Executive-Summary.pdf"),
]

def download_report(filename: str, url: str, output_dir: Path) -> bool:
    """Download a single PDF report. Returns True on success."""
    filepath = output_dir / filename
    if filepath.exists() and filepath.stat().st_size > 10_000:
        print(f"  SKIP (exists): {filename}")
        return True

    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True, allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        size = int(resp.headers.get("Content-Length", 0))

        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        actual_size = filepath.stat().st_size
        if actual_size < 5_000:
            filepath.unlink()
            print(f"  FAIL (too small {actual_size}B): {filename}")
            return False

        print(f"  OK ({actual_size / 1_048_576:.1f} MB): {filename}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  FAIL ({type(e).__name__}): {filename}")
        return False


def main():
    print(f"Downloading {len(REPORTS)} ESG reports to:\n  {OUTPUT_DIR}\n")

    success = 0
    failed = 0
    skipped = 0

    for i, (filename, url) in enumerate(REPORTS, 1):
        print(f"[{i}/{len(REPORTS)}] {filename}")
        filepath = OUTPUT_DIR / filename
        if filepath.exists() and filepath.stat().st_size > 10_000:
            skipped += 1
            print(f"  SKIP (exists)")
            continue

        if download_report(filename, url, OUTPUT_DIR):
            success += 1
        else:
            failed += 1

        # Small delay to be respectful
        time.sleep(0.5)

    print(f"\nDone! Success: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total PDFs in folder: {len(list(OUTPUT_DIR.glob('*.pdf')))}")


if __name__ == "__main__":
    main()
