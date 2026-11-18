"""Verify published field summaries from the authors' PDF, without claim to microdata."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re

from pypdf import PdfReader
import requests


URL = ("https://business.columbia.edu/sites/default/files-efs/citation_file_upload/"
       "Automating_the_B2B_Salesperson1_0.pdf")
EXPECTED = "2679ecfad967a7453f15a0cdbbaae80c3ed43e48380b9f51d554defef93ee4b0"


def audit(data):
    checksum = hashlib.sha256(data).hexdigest()
    if checksum != EXPECTED:
        raise ValueError("Author PDF has changed; check version before extracting")
    pdf = PdfReader(io.BytesIO(data))
    pages = {i: " ".join(pdf.pages[i-1].extract_text().split()) for i in (20, 23, 25)}
    m = re.search(r"GPcontrol = \$([\d.]+), GPtreatment = \$([\d.]+), Z = (\S+), p = ([\d.]+)\)", pages[23])
    control, treated = float(m[1]), float(m[2])
    z = float(m[3].replace("−", "-"))
    reported_p = float(m[4])
    new_p = math.erfc(abs(z) / math.sqrt(2))
    assert round(new_p, 3) == reported_p
    assert round(treated - control, 2) == 10.95
    compliance = re.search(r"recommendation\), was ([\d.]+)%", pages[20])
    assert compliance and all(token in pages[20] for token in ("2,075", "1,045", "4,142"))
    fitted = re.search(r"Client acceptance of price Treatment ([\d.]+)\S*\s*\(([\d.]+)\)", pages[25])
    beta, se = float(fitted[1]), float(fitted[2])
    return {"type": "published field summary audit; original row-level data unavailable here",
            "source_pdf": URL, "source_sha256": checksum, "source_bytes": len(data),
            "source_pages_one_based": list(pages), "study": "B2B aluminum retailer, not the proposed advisor",
            "business_days": 8, "quotes": 2075, "customers": 1045, "line_prices": 4142,
            "full_or_directional_compliance_percent": float(compliance[1]),
            "gross_profit_usd_per_line": {"control": control, "treatment": treated,
                "difference": round(treated - control, 2), "relative_change": treated / control - 1},
            "quote_clustered_rank_sum": {"reported_z": z, "reported_p": reported_p,
                "two_sided_p_from_reported_z": new_p},
            "table3_client_acceptance_probit": {"coefficient": beta, "se": se,
                "wald_ci95_from_rounded_numbers": [beta - 1.96 * se, beta + 1.96 * se],
                "scale": "latent index, not a probability-point or profit effect"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", help="Local archived author PDF; otherwise fetch the public URL")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.pdf:
        data = Path(args.pdf).read_bytes()
    else:
        response = requests.get(URL, timeout=45)
        response.raise_for_status()
        data = response.content
    report = json.dumps(audit(data), indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        print(report)
