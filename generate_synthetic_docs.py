"""
ALIP – Aquila Land Intelligence Platform
Synthetic Mineral Licensing Document Generator
================================================
Generates labelled PDF documents for all 8 Tier 1 document classes.
Each document includes realistic variation in applicant names, dates,
license numbers, coordinates, and jurisdiction-specific terminology.

Output structure:
    output/
      exploration_license/
      prospecting_permit/
      mining_lease/
      small_scale_mining_permit/
      new_license_application/
      renewal_application/
      environmental_impact_assessment/
      environmental_compliance_certificate/
      manifest.csv

Usage:
    python generate_synthetic_docs.py
    python generate_synthetic_docs.py --docs-per-class 25 --output ./data
"""

import os
import csv
import random
import argparse
from datetime import date, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# ── Seed for reproducibility ────────────────────────────────────────────────
random.seed(42)

# ── Variation Data ───────────────────────────────────────────────────────────

JURISDICTIONS = [
    {
        "country": "Zimbabwe",
        "ministry": "Ministry of Mines and Mining Development",
        "regulator": "Zimbabwe Mining Cadastre Office (ZMCO)",
        "act": "Mines and Minerals Act [Chapter 21:05]",
        "code": "ZW",
        "currency": "USD",
        "cities": ["Harare", "Bulawayo", "Gweru", "Mutare", "Kwekwe"],
        "regions": ["Mashonaland West", "Mashonaland Central", "Matabeleland North",
                    "Midlands", "Manicaland", "Masvingo"],
        "minister_title": "Minister of Mines and Mining Development",
        "minerals": ["gold", "platinum", "chrome", "coal", "diamonds",
                     "lithium", "nickel", "iron ore"],
    },
    {
        "country": "Zambia",
        "ministry": "Ministry of Mines and Mineral Development",
        "regulator": "Zambia Mining Cadastre Administration System (MCAS)",
        "act": "Mines and Minerals Development Act No. 11 of 2015",
        "code": "ZM",
        "currency": "USD",
        "cities": ["Lusaka", "Kitwe", "Ndola", "Livingstone", "Kabwe"],
        "regions": ["Copperbelt Province", "North-Western Province", "Lusaka Province",
                    "Southern Province", "Eastern Province"],
        "minister_title": "Minister of Mines and Mineral Development",
        "minerals": ["copper", "cobalt", "emeralds", "gold", "manganese",
                     "zinc", "lead", "uranium"],
    },
    {
        "country": "Botswana",
        "ministry": "Ministry of Mineral Resources, Green Technology and Energy Security",
        "regulator": "Department of Mines (DOM)",
        "act": "Mines and Minerals Act (Cap. 66:01)",
        "code": "BW",
        "currency": "BWP",
        "cities": ["Gaborone", "Francistown", "Maun", "Serowe", "Lobatse"],
        "regions": ["Central District", "North-East District", "Ngamiland District",
                    "Kgatleng District", "South-East District"],
        "minister_title": "Minister of Mineral Resources",
        "minerals": ["diamonds", "coal", "copper", "nickel", "gold",
                     "soda ash", "salt", "silver"],
    },
    {
        "country": "Namibia",
        "ministry": "Ministry of Mines and Energy",
        "regulator": "Geological Survey of Namibia (GSN)",
        "act": "Minerals (Prospecting and Mining) Act, Act 33 of 1992",
        "code": "NA",
        "currency": "NAD",
        "cities": ["Windhoek", "Swakopmund", "Walvis Bay", "Oshakati", "Rundu"],
        "regions": ["Erongo Region", "Khomas Region", "Otjozondjupa Region",
                    "Kunene Region", "Kavango East Region"],
        "minister_title": "Minister of Mines and Energy",
        "minerals": ["diamonds", "uranium", "gold", "zinc", "copper",
                     "lead", "silver", "coal"],
    },
]

COMPANIES = [
    "Apex Mining Corporation Ltd", "Savanna Resources (Pvt) Ltd",
    "Great Rift Minerals Inc.", "Southern Cross Mining Co. Ltd",
    "Kalahari Exploration Group", "Victoria Basin Resources Ltd",
    "Zambezi Mining Consortium", "Highveld Mineral Ventures (Pvt) Ltd",
    "Sentinel Gold & Minerals Ltd", "Mutare Mining Enterprises Ltd",
    "Copper Crown Investments Ltd", "AfriOre Exploration (Pty) Ltd",
    "Nkosi Resources Group Ltd", "Baobab Mineral Holdings Ltd",
    "Cascades Mining & Exploration Inc.", "Manica Gold Syndicate Ltd",
    "Limpopo Basin Minerals Co.", "Great Zimbabwe Mining Corp.",
    "Thornfields Exploration Ltd", "Meridian Mineral Ventures Ltd",
]

INDIVIDUALS = [
    "Mr. Tendai Moyo", "Ms. Chipo Ndlovu", "Mr. Farai Mutasa",
    "Dr. Sibongile Dube", "Mr. Tapiwa Chikwanda", "Ms. Rudo Mhanda",
    "Mr. Blessing Nyathi", "Ms. Grace Mwangi", "Mr. Prosper Chikosi",
    "Dr. Nomvula Sithole",
]

GEOLOGISTS = [
    "Dr. S. Mubaiwa (B.Sc. Geology, M.Sc. Economic Geology, Ph.D.)",
    "Mr. T. Chikwanda (B.Sc. Geology, M.Sc. Mining Engineering)",
    "Ms. R. Ndlovu (B.Sc. Geology, M.Sc. Geochemistry)",
    "Dr. P. Moyo (Ph.D. Geology, 15 years field experience)",
    "Mr. B. Ncube (B.Sc. Geology, M.Sc. Mineral Exploration)",
]

def rand_date(start_year=2020, end_year=2024):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def rand_future_date(from_date, years=2):
    return from_date + timedelta(days=365 * years + random.randint(-30, 30))

def rand_coords(jur):
    """Generate plausible bounding coordinates for the jurisdiction."""
    bounds = {
        "ZW": (-22.4, -15.6, 25.2, 33.1),
        "ZM": (-18.1, -8.2, 21.9, 33.7),
        "BW": (-26.9, -17.8, 19.9, 29.4),
        "NA": (-28.9, -16.9, 11.7, 25.3),
    }
    s, n, w, e = bounds[jur["code"]]
    lat1 = round(random.uniform(s, n - 0.5), 4)
    lat2 = round(lat1 + random.uniform(0.3, 1.5), 4)
    lon1 = round(random.uniform(w, e - 0.5), 4)
    lon2 = round(lon1 + random.uniform(0.3, 1.5), 4)
    area = round(abs((lat2 - lat1) * (lon2 - lon1)) * 12321, 1)
    return {
        "sw": f"{abs(lat1):.4f}°S, {lon1:.4f}°E",
        "nw": f"{abs(lat2):.4f}°S, {lon1:.4f}°E",
        "ne": f"{abs(lat2):.4f}°S, {lon2:.4f}°E",
        "se": f"{abs(lat1):.4f}°S, {lon2:.4f}°E",
        "area_ha": area,
    }

def license_number(prefix, jur, year):
    n = random.randint(1000, 9999)
    return f"{prefix}-{jur['code']}-{year}-{n:04d}"

def pick(lst):
    return random.choice(lst)

def pick_applicant():
    if random.random() < 0.75:
        return pick(COMPANIES), "company"
    return pick(INDIVIDUALS), "individual"

# ── Style helpers ─────────────────────────────────────────────────────────────

def get_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Normal"],
            fontSize=14, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
            fontSize=11, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=2),
        "ministry": ParagraphStyle("ministry", parent=base["Normal"],
            fontSize=10, fontName="Helvetica",
            alignment=TA_CENTER, spaceAfter=2, textColor=colors.darkblue),
        "ref": ParagraphStyle("ref", parent=base["Normal"],
            fontSize=9, fontName="Helvetica",
            alignment=TA_RIGHT, spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["Normal"],
            fontSize=10, fontName="Helvetica",
            alignment=TA_JUSTIFY, spaceAfter=8, leading=14),
        "body_bold": ParagraphStyle("body_bold", parent=base["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            spaceAfter=4),
        "small": ParagraphStyle("small", parent=base["Normal"],
            fontSize=8.5, fontName="Helvetica",
            alignment=TA_JUSTIFY, spaceAfter=6, leading=12),
        "label": ParagraphStyle("label", parent=base["Normal"],
            fontSize=9, fontName="Helvetica-Bold", spaceAfter=2),
        "field": ParagraphStyle("field", parent=base["Normal"],
            fontSize=9, fontName="Helvetica", spaceAfter=6,
            leftIndent=20),
        "section": ParagraphStyle("section", parent=base["Normal"],
            fontSize=11, fontName="Helvetica-Bold",
            spaceAfter=4, spaceBefore=10,
            textColor=colors.darkblue),
        "footer": ParagraphStyle("footer", parent=base["Normal"],
            fontSize=8, fontName="Helvetica",
            alignment=TA_CENTER, textColor=colors.grey),
        "sign": ParagraphStyle("sign", parent=base["Normal"],
            fontSize=9, fontName="Helvetica", spaceAfter=2),
        "centered": ParagraphStyle("centered", parent=base["Normal"],
            fontSize=10, fontName="Helvetica", alignment=TA_CENTER,
            spaceAfter=6),
    }
    return styles

def build_pdf(path, story):
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        rightMargin=2.5*cm, leftMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm
    )
    doc.build(story)

def hr(color=colors.darkblue, thickness=1.5):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=6, spaceBefore=4)

def sp(h=0.3):
    return Spacer(1, h*cm)

def letterhead(story, jur, s):
    story.append(Paragraph(jur["country"].upper() + " GOVERNMENT", s["ministry"]))
    story.append(Paragraph(jur["ministry"].upper(), s["title"]))
    story.append(Paragraph(jur["regulator"], s["ministry"]))
    story.append(hr())

# ── DOCUMENT GENERATORS ──────────────────────────────────────────────────────

def gen_exploration_license(path, jur, idx):
    s = get_styles()
    story = []
    applicant, atype = pick_applicant()
    issued = rand_date(2020, 2024)
    expires = rand_future_date(issued, years=random.choice([2, 3]))
    mineral = pick(jur["minerals"])
    coords = rand_coords(jur)
    lic_no = license_number("EL", jur, issued.year)
    region = pick(jur["regions"])

    letterhead(story, jur, s)
    story.append(sp(0.2))
    story.append(Paragraph("EXPLORATION LICENSE", s["subtitle"]))
    story.append(Paragraph(f"License No: <b>{lic_no}</b>", s["ref"]))
    story.append(sp(0.3))

    story.append(Paragraph(
        f"In exercise of the powers conferred by Section 36 of the "
        f"<i>{jur['act']}</i>, the {jur['minister_title']} hereby grants "
        f"an Exploration License to:", s["body"]))

    story.append(Paragraph(f"<b>{applicant}</b>", s["centered"]))
    story.append(sp(0.2))

    story.append(Paragraph(
        f"hereinafter referred to as the <b>\"License Holder\"</b>, subject to the "
        f"terms, conditions, and obligations set forth herein.", s["body"]))

    story.append(Paragraph("1. LICENSED AREA", s["section"]))
    story.append(Paragraph(
        f"The License Holder is hereby authorised to conduct geological exploration "
        f"activities within the area designated as <b>{lic_no} Block</b>, situated in "
        f"<b>{region}</b>, {jur['country']}, and bounded by the following coordinates:", s["body"]))

    coord_data = [
        ["Corner", "Latitude", "Longitude"],
        ["South-West", coords["sw"].split(",")[0], coords["sw"].split(",")[1]],
        ["North-West", coords["nw"].split(",")[0], coords["nw"].split(",")[1]],
        ["North-East", coords["ne"].split(",")[0], coords["ne"].split(",")[1]],
        ["South-East", coords["se"].split(",")[0], coords["se"].split(",")[1]],
    ]
    t = Table(coord_data, colWidths=[4*cm, 5*cm, 5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightblue]),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(sp(0.3))
    story.append(Paragraph(f"Total licensed area: approximately <b>{coords['area_ha']:,.1f} hectares</b>.", s["body"]))

    story.append(Paragraph("2. TARGET MINERALS", s["section"]))
    story.append(Paragraph(
        f"This license authorises exploration for <b>{mineral}</b> and associated minerals "
        f"as may be identified during geological survey activities.", s["body"]))

    story.append(Paragraph("3. DURATION", s["section"]))
    story.append(Paragraph(
        f"This Exploration License shall be valid from <b>{issued.strftime('%d %B %Y')}</b> "
        f"to <b>{expires.strftime('%d %B %Y')}</b>, unless earlier revoked, surrendered, "
        f"or renewed in accordance with the provisions of the Act.", s["body"]))

    story.append(Paragraph("4. CONDITIONS AND OBLIGATIONS", s["section"]))
    conditions = [
        f"The License Holder shall submit an annual work programme and expenditure report "
        f"to the {jur['regulator']} not later than 31 January of each calendar year.",
        f"Minimum annual exploration expenditure shall not be less than "
        f"{jur['currency']} {random.randint(20, 150) * 1000:,}.",
        "The License Holder shall comply with all applicable environmental laws and "
        "regulations and shall obtain all necessary environmental approvals prior to "
        "commencing any ground disturbance activities.",
        "The License Holder shall not transfer, cede, or encumber this license without "
        "the prior written consent of the Minister.",
        "The License Holder shall allow authorised officers of the Ministry to inspect "
        "the licensed area and all records relating to exploration activities at any time.",
        "Upon expiry or cancellation of this license, the License Holder shall rehabilitate "
        "all areas disturbed during exploration activities to the satisfaction of the Director of Mines.",
    ]
    for i, cond in enumerate(conditions, 1):
        story.append(Paragraph(f"{i}. {cond}", s["body"]))

    story.append(sp(0.5))
    story.append(hr(colors.grey, 0.5))
    story.append(Paragraph(
        f"Granted at {pick(jur['cities'])} this <b>{issued.strftime('%d')}th day of "
        f"{issued.strftime('%B %Y')}</b>.", s["body"]))
    story.append(sp(0.8))
    story.append(Paragraph("_______________________________", s["sign"]))
    story.append(Paragraph(f"<b>{jur['minister_title']}</b>", s["sign"]))
    story.append(Paragraph(jur["ministry"], s["sign"]))
    story.append(sp(0.3))
    story.append(Paragraph(
        f"<i>This license is issued under the authority of the {jur['act']}. "
        f"Any contravention of the conditions herein may result in suspension or "
        f"cancellation of this license.</i>", s["footer"]))

    build_pdf(path, story)
    return {"label": "exploration_license", "license_no": lic_no,
            "applicant": applicant, "jurisdiction": jur["country"],
            "issued": str(issued), "expires": str(expires), "mineral": mineral}


def gen_prospecting_permit(path, jur, idx):
    s = get_styles()
    story = []
    applicant, atype = pick_applicant()
    issued = rand_date(2020, 2024)
    expires = rand_future_date(issued, years=1)
    mineral = pick(jur["minerals"])
    coords = rand_coords(jur)
    perm_no = license_number("PP", jur, issued.year)
    region = pick(jur["regions"])

    letterhead(story, jur, s)
    story.append(sp(0.2))
    story.append(Paragraph("PROSPECTING PERMIT", s["subtitle"]))
    story.append(Paragraph(f"Permit No: <b>{perm_no}</b>", s["ref"]))
    story.append(sp(0.3))

    story.append(Paragraph(
        f"Pursuant to Section 22 of the <i>{jur['act']}</i>, this Prospecting Permit is "
        f"hereby issued to <b>{applicant}</b> (hereinafter \"the Permittee\"), "
        f"authorising initial investigation and sampling activities within the area "
        f"described below for the purpose of ascertaining the presence and extent of "
        f"<b>{mineral}</b> deposits.", s["body"]))

    story.append(Paragraph("PERMIT DETAILS", s["section"]))
    details = [
        ["Permittee", applicant],
        ["Permit Number", perm_no],
        ["Target Mineral(s)", mineral.capitalize()],
        ["Province / Region", region],
        ["Approximate Area", f"{coords['area_ha']:,.1f} hectares"],
        ["Date of Issue", issued.strftime("%d %B %Y")],
        ["Date of Expiry", expires.strftime("%d %B %Y")],
        ["Permit Duration", "12 months"],
    ]
    t = Table(details, colWidths=[5*cm, 9*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(sp(0.4))

    story.append(Paragraph("SCOPE OF PERMITTED ACTIVITIES", s["section"]))
    activities = [
        "Surface geological mapping and sampling",
        "Geochemical soil and stream sediment sampling",
        "Geophysical surveys (ground magnetic, IP, and EM surveys)",
        "Shallow trenching not exceeding 3 metres in depth",
        "Collection of rock chip, grab, and channel samples for laboratory analysis",
        "Installation of temporary survey pegs and markers",
    ]
    for act in activities:
        story.append(Paragraph(f"• {act}", s["body"]))

    story.append(Paragraph("TERMS AND CONDITIONS", s["section"]))
    story.append(Paragraph(
        "This permit is issued subject to the following conditions:", s["body"]))
    terms = [
        "The Permittee shall not conduct any drilling or bulk sampling without prior written approval.",
        f"A completion report shall be submitted to the {jur['regulator']} within 30 days of permit expiry.",
        "All activities shall comply with applicable environmental and safety legislation.",
        "This permit is non-transferable and personal to the Permittee.",
        f"Permit fee of {jur['currency']} {random.randint(500, 5000):,} has been paid and acknowledged.",
    ]
    for i, term in enumerate(terms, 1):
        story.append(Paragraph(f"{i}. {term}", s["small"]))

    story.append(sp(0.6))
    story.append(hr(colors.grey, 0.5))
    story.append(Paragraph(
        f"Issued by authority of the {jur['minister_title']} at "
        f"{pick(jur['cities'])}, {issued.strftime('%d %B %Y')}.", s["body"]))
    story.append(sp(0.6))
    story.append(Paragraph("_______________________________", s["sign"]))
    story.append(Paragraph("<b>Director of Mines</b>", s["sign"]))
    story.append(Paragraph(f"On behalf of the {jur['minister_title']}", s["sign"]))

    build_pdf(path, story)
    return {"label": "prospecting_permit", "permit_no": perm_no,
            "applicant": applicant, "jurisdiction": jur["country"],
            "issued": str(issued), "expires": str(expires), "mineral": mineral}


def gen_mining_lease(path, jur, idx):
    s = get_styles()
    story = []
    applicant, _ = pick_applicant()
    issued = rand_date(2018, 2023)
    duration_years = random.choice([10, 15, 20, 25])
    expires = rand_future_date(issued, years=duration_years)
    mineral = pick(jur["minerals"])
    coords = rand_coords(jur)
    lease_no = license_number("ML", jur, issued.year)
    region = pick(jur["regions"])
    royalty_rate = round(random.uniform(2.0, 7.5), 1)
    production_cap = random.randint(50000, 500000)

    letterhead(story, jur, s)
    story.append(sp(0.2))
    story.append(Paragraph("MINING LEASE", s["subtitle"]))
    story.append(Paragraph(f"Lease No: <b>{lease_no}</b>", s["ref"]))
    story.append(sp(0.2))
    story.append(hr(colors.darkblue, 2))

    story.append(Paragraph(
        f"THIS MINING LEASE is granted by the Government of the Republic of "
        f"{jur['country']}, acting through the {jur['minister_title']}, pursuant to "
        f"Section 52 of the <i>{jur['act']}</i>, to:", s["body"]))
    story.append(Paragraph(f"<b>{applicant}</b>", s["centered"]))
    story.append(Paragraph(
        f"(hereinafter referred to as the <b>\"Lessee\"</b>), for the purpose of "
        f"mining, extracting, processing, and disposing of <b>{mineral}</b> and "
        f"associated minerals from the area described herein.", s["body"]))

    story.append(Paragraph("ARTICLE 1 — DESCRIPTION OF LEASE AREA", s["section"]))
    story.append(Paragraph(
        f"The lease area, designated as <b>{lease_no} Mining Block</b>, is situated "
        f"in the <b>{region}</b> of {jur['country']} and encompasses approximately "
        f"<b>{coords['area_ha']:,.1f} hectares</b>, bounded by the coordinates set "
        f"out in Schedule A attached hereto.", s["body"]))

    story.append(Paragraph("ARTICLE 2 — DURATION", s["section"]))
    story.append(Paragraph(
        f"This Mining Lease shall be in force for a period of <b>{duration_years} years</b> "
        f"commencing on <b>{issued.strftime('%d %B %Y')}</b> and expiring on "
        f"<b>{expires.strftime('%d %B %Y')}</b>, unless sooner determined in accordance "
        f"with the provisions of this Lease or the Act.", s["body"]))

    story.append(Paragraph("ARTICLE 3 — ROYALTIES AND FINANCIAL OBLIGATIONS", s["section"]))
    story.append(Paragraph(
        f"The Lessee shall pay royalties to the Government at the rate of "
        f"<b>{royalty_rate}%</b> of the gross value of all minerals produced and "
        f"sold from the lease area. Royalty payments shall be made quarterly. "
        f"Annual production is estimated not to exceed "
        f"<b>{production_cap:,} tonnes</b> of ore processed.", s["body"]))

    story.append(Paragraph("ARTICLE 4 — ENVIRONMENTAL OBLIGATIONS", s["section"]))
    story.append(Paragraph(
        "The Lessee shall, prior to commencement of mining operations, obtain all "
        "necessary environmental approvals including an approved Environmental Impact "
        "Assessment (EIA) and Environmental Management Plan (EMP). A rehabilitation "
        "bond of not less than the estimated cost of full site rehabilitation shall "
        "be deposited with the Ministry before operations commence.", s["body"]))

    story.append(Paragraph("ARTICLE 5 — REPORTING OBLIGATIONS", s["section"]))
    reports = [
        "Monthly production returns submitted by the 15th of the following month.",
        "Quarterly royalty statements and payments.",
        "Annual environmental compliance report.",
        "Immediate notification of any significant environmental incident.",
        "Annual audit of rehabilitation fund and progress report.",
    ]
    for r in reports:
        story.append(Paragraph(f"• {r}", s["body"]))

    story.append(Paragraph("ARTICLE 6 — BREACH AND TERMINATION", s["section"]))
    story.append(Paragraph(
        "In the event of material breach of any condition of this Lease, the Minister "
        "may, after giving 60 days written notice and opportunity to remedy, suspend or "
        "cancel this Lease. The Lessee shall have the right to appeal any such decision "
        f"to the Mining Appeals Tribunal within 30 days of receipt of notice.", s["body"]))

    story.append(sp(0.5))
    story.append(hr())
    story.append(Paragraph("IN WITNESS WHEREOF", s["body_bold"]))
    story.append(Paragraph(
        f"This Mining Lease has been executed at {pick(jur['cities'])} on "
        f"the {issued.strftime('%d')}th day of {issued.strftime('%B %Y')}.", s["body"]))
    story.append(sp(0.8))
    tbl = Table([
        [Paragraph("_____________________\n<b>LESSEE</b>", s["sign"]),
         Paragraph(f"_____________________\n<b>{jur['minister_title'].upper()}</b>", s["sign"])]
    ], colWidths=[7*cm, 7*cm])
    story.append(tbl)

    build_pdf(path, story)
    return {"label": "mining_lease", "lease_no": lease_no,
            "applicant": applicant, "jurisdiction": jur["country"],
            "issued": str(issued), "expires": str(expires),
            "mineral": mineral, "royalty_rate": royalty_rate}


def gen_small_scale_permit(path, jur, idx):
    s = get_styles()
    story = []
    applicant, atype = pick_applicant()
    issued = rand_date(2021, 2024)
    expires = rand_future_date(issued, years=2)
    mineral = pick(jur["minerals"])
    coords = rand_coords(jur)
    perm_no = license_number("SSMP", jur, issued.year)
    region = pick(jur["regions"])

    letterhead(story, jur, s)
    story.append(sp(0.2))
    story.append(Paragraph("SMALL-SCALE MINING PERMIT", s["subtitle"]))
    story.append(Paragraph(f"Permit No: <b>{perm_no}</b>", s["ref"]))
    story.append(sp(0.3))

    story.append(Paragraph(
        f"This Small-Scale Mining Permit is issued under Section 44 of the "
        f"<i>{jur['act']}</i> to:", s["body"]))
    story.append(Paragraph(f"<b>{applicant}</b>", s["centered"]))

    story.append(Paragraph("PERMIT DETAILS", s["section"]))
    details = [
        ["Permit Holder", applicant],
        ["Permit Number", perm_no],
        ["Category", "Small-Scale Mining"],
        ["Target Mineral", mineral.capitalize()],
        ["Region", region],
        ["Licensed Area", f"{min(coords['area_ha'], 100.0):,.1f} hectares (max. 100 ha)"],
        ["Effective Date", issued.strftime("%d %B %Y")],
        ["Expiry Date", expires.strftime("%d %B %Y")],
        ["Max. Monthly Production", f"{random.randint(5, 50)} tonnes"],
        ["Permit Fee Paid", f"{jur['currency']} {random.randint(200, 1500):,}"],
    ]
    t = Table(details, colWidths=[5*cm, 9*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(sp(0.4))

    story.append(Paragraph("PERMITTED OPERATIONS", s["section"]))
    story.append(Paragraph(
        "The permit holder is authorised to conduct the following small-scale "
        "mining operations within the licensed area:", s["body"]))
    ops = [
        f"Surface and shallow underground extraction of {mineral} ore to a maximum depth of {random.randint(15,50)} metres.",
        "Crushing, screening, and gravity concentration of ore on site.",
        "Sale of mineral output to licensed mineral buyers or processing plants only.",
        "Employment of not more than " + str(random.randint(10, 50)) + " workers on site at any time.",
    ]
    for op in ops:
        story.append(Paragraph(f"• {op}", s["body"]))

    story.append(Paragraph("HEALTH, SAFETY AND ENVIRONMENT", s["section"]))
    story.append(Paragraph(
        "The permit holder shall comply with all occupational health and safety "
        "regulations applicable to small-scale mining operations. An environmental "
        "management checklist shall be maintained on site and made available to "
        "inspectors upon request. No use of mercury or other hazardous substances "
        "is permitted without prior written authorisation.", s["body"]))

    story.append(sp(0.6))
    story.append(hr(colors.grey, 0.5))
    story.append(Paragraph(
        f"Issued at {pick(jur['cities'])} on {issued.strftime('%d %B %Y')}.", s["body"]))
    story.append(sp(0.5))
    story.append(Paragraph("_______________________________", s["sign"]))
    story.append(Paragraph("<b>Provincial Mining Director</b>", s["sign"]))
    story.append(Paragraph(jur["regulator"], s["sign"]))

    build_pdf(path, story)
    return {"label": "small_scale_mining_permit", "permit_no": perm_no,
            "applicant": applicant, "jurisdiction": jur["country"],
            "issued": str(issued), "expires": str(expires), "mineral": mineral}


def gen_new_application(path, jur, idx):
    s = get_styles()
    story = []
    applicant, atype = pick_applicant()
    submitted = rand_date(2021, 2024)
    mineral = pick(jur["minerals"])
    coords = rand_coords(jur)
    ref_no = f"APP-{jur['code']}-{submitted.year}-{random.randint(1000,9999)}"
    region = pick(jur["regions"])
    lic_type = pick(["Exploration License", "Mining Lease", "Prospecting Permit"])
    geologist = pick(GEOLOGISTS)

    letterhead(story, jur, s)
    story.append(sp(0.2))
    story.append(Paragraph(f"APPLICATION FOR {lic_type.upper()}", s["subtitle"]))
    story.append(Paragraph(f"Reference: <b>{ref_no}</b>   Date: {submitted.strftime('%d %B %Y')}", s["ref"]))
    story.append(sp(0.2))
    story.append(Paragraph(f"To: The {jur['minister_title']}", s["body_bold"]))
    story.append(Paragraph(jur["ministry"], s["body"]))
    story.append(sp(0.2))

    story.append(Paragraph(
        f"I/We, <b>{applicant}</b>, hereby apply for a {lic_type} over the area "
        f"described herein, pursuant to the provisions of the <i>{jur['act']}</i>.", s["body"]))

    story.append(Paragraph("SECTION A — APPLICANT DETAILS", s["section"]))
    app_details = [
        ["Full Name / Company Name", applicant],
        ["Registration Number", f"{'C' if atype=='company' else 'ID'}-{random.randint(100000,999999)}"],
        ["Registered Address", f"{random.randint(1,200)} {pick(['Main','Industrial','Commercial','Mines'])} Street, {pick(jur['cities'])}, {jur['country']}"],
        ["Contact Person", pick(INDIVIDUALS).replace("Mr. ", "").replace("Ms. ", "").replace("Dr. ", "")],
        ["Telephone", f"+{random.randint(200,299)}-{random.randint(10,99)}-{random.randint(1000000,9999999)}"],
        ["Email", f"info@{applicant.lower().split()[0]}.com"],
        ["Nationality / Country of Inc.", jur["country"] if random.random() > 0.3 else pick(["South Africa", "Australia", "Canada", "UK"])],
    ]
    t = Table(app_details, colWidths=[5.5*cm, 8.5*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(sp(0.3))

    story.append(Paragraph("SECTION B — AREA DESCRIPTION", s["section"]))
    area_details = [
        ["Province / Region", region],
        ["Nearest Town / Village", pick(jur["cities"])],
        ["Approximate Area", f"{coords['area_ha']:,.1f} hectares"],
        ["South-West Corner", coords["sw"]],
        ["North-East Corner", coords["ne"]],
    ]
    t2 = Table(area_details, colWidths=[5.5*cm, 8.5*cm])
    t2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t2)
    story.append(sp(0.3))

    story.append(Paragraph("SECTION C — TARGET MINERALS AND TECHNICAL JUSTIFICATION", s["section"]))
    story.append(Paragraph(
        f"The applicant seeks to explore for <b>{mineral}</b>. Preliminary desktop "
        f"studies including analysis of airborne geophysical data, regional geological "
        f"mapping, and satellite imagery interpretation indicate anomalous signatures "
        f"consistent with {mineral} mineralisation within the application area. "
        f"The proposed technical programme will be supervised by {geologist}.", s["body"]))

    story.append(Paragraph("SECTION D — PROPOSED WORK PROGRAMME (YEAR 1)", s["section"]))
    work = [
        ["Activity", "Duration", "Estimated Cost (" + jur["currency"] + ")"],
        ["Geological mapping (1:10,000 scale)", "3 months", f"{random.randint(10,50)*1000:,}"],
        ["Geochemical soil sampling (500 samples)", "2 months", f"{random.randint(15,60)*1000:,}"],
        ["Ground geophysical survey", "2 months", f"{random.randint(20,80)*1000:,}"],
        ["Core drilling (1,500 metres)", "4 months", f"{random.randint(80,200)*1000:,}"],
        ["Laboratory assays and data analysis", "3 months", f"{random.randint(10,40)*1000:,}"],
        ["Report preparation and submission", "1 month", f"{random.randint(5,15)*1000:,}"],
    ]
    t3 = Table(work, colWidths=[7*cm, 3*cm, 4*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t3)
    story.append(sp(0.3))

    story.append(Paragraph("DECLARATION", s["section"]))
    story.append(Paragraph(
        "I/We declare that the information provided in this application is true and "
        "correct to the best of my/our knowledge. I/We undertake to comply with all "
        "conditions attached to any license or permit that may be granted pursuant to "
        "this application.", s["body"]))
    story.append(sp(0.6))
    story.append(Paragraph(f"Signed: ___________________________   Date: {submitted.strftime('%d %B %Y')}", s["sign"]))
    story.append(Paragraph(f"Name: <b>{applicant}</b>", s["sign"]))

    build_pdf(path, story)
    return {"label": "new_license_application", "ref_no": ref_no,
            "applicant": applicant, "jurisdiction": jur["country"],
            "submitted": str(submitted), "mineral": mineral,
            "license_type_applied": lic_type}


def gen_renewal_application(path, jur, idx):
    s = get_styles()
    story = []
    applicant, _ = pick_applicant()
    submitted = rand_date(2021, 2024)
    original_issue = rand_date(2018, 2021)
    original_expiry = rand_future_date(original_issue, years=random.choice([2, 3]))
    mineral = pick(jur["minerals"])
    lic_types = ["Exploration License", "Mining Lease", "Prospecting Permit"]
    lic_type = pick(lic_types)
    prefix_map = {"Exploration License": "EL", "Mining Lease": "ML", "Prospecting Permit": "PP"}
    existing_no = license_number(prefix_map[lic_type], jur, original_issue.year)
    renewal_no = f"REN-{jur['code']}-{submitted.year}-{random.randint(1000,9999)}"
    renewal_period = random.choice([2, 3])
    region = pick(jur["regions"])
    expenditure = random.randint(50, 500) * 1000

    letterhead(story, jur, s)
    story.append(sp(0.2))
    story.append(Paragraph(f"APPLICATION FOR RENEWAL OF {lic_type.upper()}", s["subtitle"]))
    story.append(Paragraph(f"Renewal Ref: <b>{renewal_no}</b>   Existing License: <b>{existing_no}</b>", s["ref"]))
    story.append(sp(0.3))

    story.append(Paragraph(
        f"To: The {jur['minister_title']}<br/>{jur['ministry']}", s["body_bold"]))
    story.append(sp(0.2))
    story.append(Paragraph(
        f"I/We, <b>{applicant}</b>, hereby apply for the renewal of {lic_type} "
        f"No. <b>{existing_no}</b> over the area situated in <b>{region}</b>, "
        f"{jur['country']}, for a further period of <b>{renewal_period} years</b>, "
        f"pursuant to Section 38 of the <i>{jur['act']}</i>.", s["body"]))

    story.append(Paragraph("1. EXISTING LICENSE DETAILS", s["section"]))
    existing = [
        ["License / Permit Number", existing_no],
        ["License Type", lic_type],
        ["Original Issue Date", original_issue.strftime("%d %B %Y")],
        ["Current Expiry Date", original_expiry.strftime("%d %B %Y")],
        ["Target Mineral(s)", mineral.capitalize()],
        ["Licensed Region", region],
    ]
    t = Table(existing, colWidths=[5.5*cm, 8.5*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(sp(0.3))

    story.append(Paragraph("2. WORK COMPLETED DURING CURRENT LICENSE PERIOD", s["section"]))
    story.append(Paragraph(
        f"During the current license period, the applicant has undertaken the following "
        f"exploration activities and incurred total expenditure of approximately "
        f"<b>{jur['currency']} {expenditure:,}</b>:", s["body"]))
    completed = [
        f"Regional geological mapping covering {random.randint(50,200)} km² at 1:25,000 scale.",
        f"Geochemical sampling programme: {random.randint(200,1500)} soil samples and {random.randint(50,200)} rock chip samples collected.",
        f"Ground geophysical surveys: {random.randint(50,300)} line-km of magnetic and IP data acquired.",
        f"Drilling programme: {random.randint(5,30)} holes totalling {random.randint(500,5000)} metres drilled.",
        f"Laboratory assays: {random.randint(100,2000)} samples submitted to certified laboratory.",
        f"Delineation of {random.randint(1,5)} target zone(s) warranting further detailed investigation.",
    ]
    for item in completed:
        story.append(Paragraph(f"• {item}", s["body"]))

    story.append(Paragraph("3. JUSTIFICATION FOR RENEWAL", s["section"]))
    story.append(Paragraph(
        f"Preliminary results indicate significant {mineral} mineralisation within the "
        f"license area. Assay results from the most recent drilling programme returned "
        f"values of up to {round(random.uniform(1.5, 15.0), 2)}g/t {mineral} over "
        f"significant intervals, confirming the prospectivity of the area. "
        f"Further work is required to fully evaluate the resource potential, "
        f"necessitating the renewal of the license for a further {renewal_period} years.", s["body"]))

    story.append(Paragraph("4. PROPOSED WORK PROGRAMME FOR RENEWAL PERIOD", s["section"]))
    story.append(Paragraph(
        f"Should renewal be granted, the applicant proposes to undertake the following "
        f"programme with a total budget of {jur['currency']} "
        f"{random.randint(200, 800) * 1000:,}:", s["body"]))
    proposed = [
        "Infill drilling to upgrade inferred resources to indicated category.",
        "Metallurgical testwork and preliminary process design.",
        "Pre-feasibility study preparation.",
        "Environmental and social baseline studies.",
        "Community consultation programme.",
    ]
    for item in proposed:
        story.append(Paragraph(f"• {item}", s["body"]))

    story.append(Paragraph("5. COMPLIANCE DECLARATION", s["section"]))
    story.append(Paragraph(
        "The applicant declares that all conditions attached to the existing license have "
        "been complied with, all fees and charges are fully paid, and all required reports "
        "have been submitted timeously. No adverse environmental incidents have occurred "
        "during the current license period.", s["body"]))

    story.append(sp(0.5))
    story.append(hr(colors.grey, 0.5))
    story.append(Paragraph(
        f"Submitted at {pick(jur['cities'])} on {submitted.strftime('%d %B %Y')}.", s["body"]))
    story.append(sp(0.5))
    story.append(Paragraph("_______________________________", s["sign"]))
    story.append(Paragraph(f"<b>{applicant}</b>", s["sign"]))
    story.append(Paragraph("Authorised Signatory", s["sign"]))

    build_pdf(path, story)
    return {"label": "renewal_application", "renewal_no": renewal_no,
            "existing_license": existing_no, "applicant": applicant,
            "jurisdiction": jur["country"], "submitted": str(submitted),
            "mineral": mineral, "renewal_years": renewal_period}


def gen_eia(path, jur, idx):
    s = get_styles()
    story = []
    applicant, _ = pick_applicant()
    submitted = rand_date(2020, 2024)
    mineral = pick(jur["minerals"])
    project_name = f"{pick(jur['regions']).split()[0]} {mineral.capitalize()} {pick(['Project','Mine','Operations'])}"
    eia_no = f"EIA-{jur['code']}-{submitted.year}-{random.randint(100,999)}"
    region = pick(jur["regions"])
    consultant = pick(COMPANIES).replace("Mining", "Environmental").replace("Resources", "Consulting")

    letterhead(story, jur, s)
    story.append(sp(0.2))
    story.append(Paragraph("ENVIRONMENTAL IMPACT ASSESSMENT REPORT", s["subtitle"]))
    story.append(Paragraph(f"EIA Reference: <b>{eia_no}</b>", s["ref"]))
    story.append(sp(0.1))
    story.append(Paragraph(f"<b>Project:</b> {project_name}", s["centered"]))
    story.append(Paragraph(f"<b>Proponent:</b> {applicant}", s["centered"]))
    story.append(Paragraph(
        f"<b>Environmental Consultant:</b> {consultant}", s["centered"]))
    story.append(Paragraph(
        f"<b>Date of Submission:</b> {submitted.strftime('%B %Y')}", s["centered"]))
    story.append(sp(0.2))
    story.append(hr())

    story.append(Paragraph("EXECUTIVE SUMMARY", s["section"]))
    story.append(Paragraph(
        f"{applicant} proposes to develop the {project_name} in <b>{region}</b>, "
        f"{jur['country']}, involving the extraction and processing of "
        f"<b>{mineral}</b> ore. This Environmental Impact Assessment (EIA) has been "
        f"prepared in accordance with the requirements of the Environmental Management "
        f"Act and the <i>{jur['act']}</i>. The assessment identifies potential "
        f"environmental impacts, evaluates their significance, and proposes mitigation "
        f"measures to ensure the project complies with national environmental standards.", s["body"]))

    story.append(Paragraph("1. PROJECT DESCRIPTION", s["section"]))
    capacity = random.randint(100000, 2000000)
    story.append(Paragraph(
        f"The proposed project entails open-pit and/or underground mining of {mineral} "
        f"ore at a nominal throughput of {capacity:,} tonnes per annum. The project "
        f"footprint covers approximately {random.randint(50,500)} hectares and includes "
        f"a processing plant, tailings storage facility, water management infrastructure, "
        f"haul roads, and ancillary facilities. The project life is estimated at "
        f"{random.randint(10,25)} years.", s["body"]))

    story.append(Paragraph("2. BASELINE ENVIRONMENTAL CONDITIONS", s["section"]))
    baseline_topics = [
        ("2.1 Topography and Soils",
         f"The project area is characterised by {pick(['undulating','flat','hilly','gently sloping'])} "
         f"terrain with {pick(['sandy loam','clay','sandy','loamy'])} soils of {pick(['low','moderate','high'])} "
         f"agricultural potential."),
        ("2.2 Hydrology and Water Resources",
         f"The area falls within the {pick(['Zambezi','Limpopo','Save','Kafue'])} catchment. "
         f"Seasonal streams are present and the water table depth ranges from "
         f"{random.randint(5,20)} to {random.randint(25,60)} metres below surface."),
        ("2.3 Vegetation and Biodiversity",
         f"The vegetation is predominantly {pick(['miombo woodland','mopane woodland','mixed bushveld','grassland'])}. "
         f"No critically endangered species have been identified within the project footprint."),
        ("2.4 Socio-Economic Environment",
         f"The nearest community is located {random.randint(2,25)} km from the proposed mine. "
         f"The project is expected to create {random.randint(50,500)} direct employment "
         f"opportunities during construction and {random.randint(30,300)} permanent positions during operations."),
    ]
    for title, content in baseline_topics:
        story.append(Paragraph(title, s["body_bold"]))
        story.append(Paragraph(content, s["body"]))

    story.append(Paragraph("3. IDENTIFICATION AND ASSESSMENT OF IMPACTS", s["section"]))
    impacts = [
        ["Impact", "Phase", "Significance", "Mitigation"],
        ["Topsoil loss and erosion", "Construction", "Moderate", "Topsoil stockpiling; progressive rehabilitation"],
        ["Dust generation", "Operations", "Moderate-High", "Water suppression; enclosed conveyors"],
        ["Groundwater contamination", "Operations", "High", "Lined tailings facility; monitoring boreholes"],
        ["Noise and vibration", "Operations", "Low-Moderate", "Blast scheduling; buffer zones"],
        ["Vegetation clearance", "Construction", "Moderate", "Minimise footprint; offset planting"],
        ["Surface water quality", "Operations", "High", "Diversion channels; water treatment plant"],
        ["Community displacement", "Pre-construction", "Low", "Voluntary resettlement with compensation"],
        ["Waste rock and tailings", "Operations", "High", "Geotechnically stable TSF; ARD prevention"],
    ]
    t = Table(impacts, colWidths=[4.5*cm, 2.5*cm, 3*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(sp(0.3))

    story.append(Paragraph("4. ENVIRONMENTAL MANAGEMENT PLAN SUMMARY", s["section"]))
    story.append(Paragraph(
        "A detailed Environmental Management Plan (EMP) accompanies this EIA as "
        "Appendix D. The EMP sets out specific mitigation measures, monitoring "
        "programmes, responsible parties, and timeframes for all identified impacts. "
        "An annual environmental audit will be conducted by an independent auditor "
        f"and submitted to {jur['regulator']} by 31 March of each year.", s["body"]))

    story.append(Paragraph("5. CONCLUSION AND RECOMMENDATION", s["section"]))
    story.append(Paragraph(
        f"Subject to implementation of the mitigation measures detailed in the EMP, "
        f"the {project_name} is considered environmentally acceptable. The environmental "
        f"consultant recommends approval of the project subject to conditions.", s["body"]))

    story.append(sp(0.4))
    story.append(hr(colors.grey, 0.5))
    story.append(Paragraph(
        f"Prepared by: <b>{consultant}</b> | EIA No: {eia_no} | {submitted.strftime('%B %Y')}", s["footer"]))

    build_pdf(path, story)
    return {"label": "environmental_impact_assessment", "eia_no": eia_no,
            "project_name": project_name, "applicant": applicant,
            "jurisdiction": jur["country"], "submitted": str(submitted),
            "mineral": mineral}


def gen_env_compliance_cert(path, jur, idx):
    s = get_styles()
    story = []
    applicant, _ = pick_applicant()
    issued = rand_date(2021, 2024)
    expires = rand_future_date(issued, years=5)
    mineral = pick(jur["minerals"])
    project_name = f"{pick(jur['regions']).split()[0]} {mineral.capitalize()} {pick(['Project','Mine'])}"
    cert_no = f"ECC-{jur['code']}-{issued.year}-{random.randint(100,999)}"
    eia_ref = f"EIA-{jur['code']}-{issued.year - random.randint(0,1)}-{random.randint(100,999)}"
    region = pick(jur["regions"])

    letterhead(story, jur, s)
    story.append(sp(0.3))
    story.append(Paragraph("ENVIRONMENTAL COMPLIANCE CERTIFICATE", s["subtitle"]))
    story.append(sp(0.2))
    story.append(hr(colors.darkblue, 2))
    story.append(sp(0.2))
    story.append(Paragraph(f"Certificate No: <b>{cert_no}</b>", s["ref"]))
    story.append(sp(0.3))

    story.append(Paragraph(
        f"The {jur['ministry']}, acting in accordance with the provisions of the "
        f"Environmental Management Act and the <i>{jur['act']}</i>, hereby certifies "
        f"that the Environmental Impact Assessment requirements for the project "
        f"described herein have been satisfied.", s["body"]))

    story.append(sp(0.2))
    story.append(hr(colors.lightblue, 0.5))
    story.append(sp(0.2))

    cert_details = [
        ["Certificate Number", cert_no],
        ["Certificate Holder", applicant],
        ["Project Name", project_name],
        ["Project Location", f"{region}, {jur['country']}"],
        ["Target Mineral(s)", mineral.capitalize()],
        ["EIA Reference Number", eia_ref],
        ["Date of Issue", issued.strftime("%d %B %Y")],
        ["Valid Until", expires.strftime("%d %B %Y")],
    ]
    t = Table(cert_details, colWidths=[5.5*cm, 8.5*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#EAF3FB"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(sp(0.4))

    story.append(Paragraph("CONDITIONS OF THIS CERTIFICATE", s["section"]))
    conditions = [
        f"This certificate is granted subject to full implementation of all mitigation "
        f"measures specified in the approved Environmental Management Plan referenced "
        f"under EIA No. {eia_ref}.",
        "The certificate holder shall conduct annual environmental audits and submit "
        f"audit reports to {jur['regulator']} by 31 March of each calendar year.",
        "Any material change to the project scope, design, or operating parameters "
        "shall require prior written approval and may necessitate a revised EIA.",
        "The certificate holder shall maintain an environmental management system "
        "certified to ISO 14001 or equivalent national standard.",
        "Community liaison meetings shall be conducted at minimum quarterly and "
        "records maintained for inspection.",
        "This certificate shall be displayed prominently at the project site and "
        "made available to inspectors upon request.",
        f"Failure to comply with any condition of this certificate may result in "
        f"suspension or revocation pursuant to applicable legislation.",
    ]
    for i, cond in enumerate(conditions, 1):
        story.append(Paragraph(f"{i}. {cond}", s["body"]))

    story.append(sp(0.4))
    story.append(hr())
    story.append(Paragraph(
        f"Issued at {pick(jur['cities'])} this <b>{issued.strftime('%d')}th day of "
        f"{issued.strftime('%B %Y')}</b>.", s["body"]))
    story.append(sp(1.0))

    sig_table = Table([
        [Paragraph("_______________________________", s["sign"]),
         Paragraph("_______________________________", s["sign"])],
        [Paragraph(f"<b>Director of Environmental Affairs</b>", s["sign"]),
         Paragraph(f"<b>{jur['minister_title']}</b>", s["sign"])],
        [Paragraph(jur["ministry"], s["sign"]),
         Paragraph(jur["ministry"], s["sign"])],
    ], colWidths=[7*cm, 7*cm])
    story.append(sig_table)
    story.append(sp(0.3))
    story.append(hr(colors.grey, 0.5))
    story.append(Paragraph(
        f"<i>This certificate is issued under the authority of the {jur['act']}. "
        f"This document is a legally binding instrument. Certificate No. {cert_no}.</i>", s["footer"]))

    build_pdf(path, story)
    return {"label": "environmental_compliance_certificate", "cert_no": cert_no,
            "project_name": project_name, "applicant": applicant,
            "jurisdiction": jur["country"], "issued": str(issued),
            "expires": str(expires), "eia_ref": eia_ref, "mineral": mineral}


# ── CLASS REGISTRY ────────────────────────────────────────────────────────────

CLASS_GENERATORS = {
    "exploration_license":               gen_exploration_license,
    "prospecting_permit":                gen_prospecting_permit,
    "mining_lease":                      gen_mining_lease,
    "small_scale_mining_permit":         gen_small_scale_permit,
    "new_license_application":           gen_new_application,
    "renewal_application":               gen_renewal_application,
    "environmental_impact_assessment":   gen_eia,
    "environmental_compliance_certificate": gen_env_compliance_cert,
}

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ALIP Synthetic Document Generator")
    parser.add_argument("--docs-per-class", type=int, default=20,
                        help="Number of documents per class (default: 20)")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory (default: ./output)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    manifest_rows = []

    print(f"\n{'='*60}")
    print(f"  ALIP Synthetic Document Generator")
    print(f"  Classes: {len(CLASS_GENERATORS)} | Docs per class: {args.docs_per_class}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    for class_name, generator_fn in CLASS_GENERATORS.items():
        class_dir = os.path.join(args.output, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"  Generating [{class_name}]")

        for i in range(args.docs_per_class):
            jur = pick(JURISDICTIONS)
            filename = f"{class_name}_{i+1:03d}.pdf"
            filepath = os.path.join(class_dir, filename)
            rel_path = os.path.join(class_name, filename)

            try:
                meta = generator_fn(filepath, jur, i)
                row = {
                    "file_path":    rel_path,
                    "label":        class_name,
                    "jurisdiction": jur["country"],
                    "filename":     filename,
                }
                row.update({k: v for k, v in meta.items()
                             if k not in ("label", "jurisdiction")})
                manifest_rows.append(row)
                print(f"    [{i+1:02d}/{args.docs_per_class}] {filename}  "
                      f"({jur['country']})", end="\r")
            except Exception as e:
                print(f"\n    ERROR generating {filename}: {e}")

        print(f"    {args.docs_per_class}/{args.docs_per_class} documents generated.    ")

    # Write manifest CSV
    manifest_path = os.path.join(args.output, "manifest.csv")
    all_keys = ["file_path", "label", "jurisdiction", "filename"]
    extra_keys = []
    for row in manifest_rows:
        for k in row:
            if k not in all_keys and k not in extra_keys:
                extra_keys.append(k)
    fieldnames = all_keys + extra_keys

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)

    total = len(manifest_rows)
    print(f"\n{'='*60}")
    print(f"  Done! {total} documents generated across {len(CLASS_GENERATORS)} classes.")
    print(f"  Manifest saved: {manifest_path}")
    print(f"{'='*60}\n")
    print("  Class distribution:")
    from collections import Counter
    counts = Counter(r["label"] for r in manifest_rows)
    for cls, cnt in counts.items():
        print(f"    {cls:<45} {cnt} docs")
    print()

if __name__ == "__main__":
    main()
