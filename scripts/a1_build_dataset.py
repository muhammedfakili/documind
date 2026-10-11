"""A1: generate the binary part of the Lumora Audio knowledge base.

The Markdown documents in data/raw were written by hand. This script produces
the formats a real client would also hand us -- PDF manuals, Excel sheets and
CSV exports -- so the loaders in A2 have to deal with all of them.

Everything here is synthetic (Lumora Audio does not exist). Several traps are
planted on purpose; they are listed in dersler/01-veri-seti.md.

Usage (from the project root):  uv run python -m scripts.a1_build_dataset
"""

from __future__ import annotations

import csv
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace
from openpyxl import Workbook
from openpyxl.styles import Font

from app.config import DATA_DIR

RAW = DATA_DIR / "raw"

# ---------------------------------------------------------------------------
# PDF manuals and terms
# ---------------------------------------------------------------------------
# Each document is a list of blocks: ("h", title) | ("p", text) | ("table", rows)
# | ("page",) for a page break. Core PDF fonts are Latin-1 only, so no euro sign
# or typographic dashes in these texts.

PULSE_MANUAL = [
    ("h", "Lumora Pulse - User Manual"),
    ("p", "Model: LA-PL-100. Document version 3.1, February 2026."),
    ("h", "1. In the box"),
    ("p", "Two earbuds, charging case, USB-C cable (50 cm), ear tips in sizes S, M and L (M is pre-installed), quick start card."),
    ("h", "2. Pairing"),
    ("p", "Open the charging case with both earbuds inside. The LED on the case blinks white: the earbuds are in pairing mode. Select 'Lumora Pulse' in your phone's Bluetooth menu. If the LED does not blink, press and hold the button on the back of the case for 3 seconds."),
    ("h", "3. Touch controls"),
    ("table", [
        ["Action", "Left earbud", "Right earbud"],
        ["Single tap", "Play / pause", "Play / pause"],
        ["Double tap", "Previous track", "Next track"],
        ["Triple tap", "Game Mode on/off (fw 3.2.0+)", "Voice assistant"],
        ["Hold 2 s", "Volume down", "Volume up"],
    ]),
    ("page",),
    ("h", "4. Battery and charging"),
    ("p", "Earbuds: up to 7 hours of playback per charge. The case holds 3 additional full charges, for up to 28 hours in total. A 10-minute charge gives 1.5 hours of playback. Full charge of the case takes about 2 hours with USB-C; wireless (Qi) charging is not supported on Pulse."),
    ("p", "Charge between 5 C and 40 C. Charging outside this range is paused automatically to protect the battery."),
    ("h", "5. Water resistance"),
    ("p", "Pulse earbuds are rated IPX4: resistant to sweat and splashes. They are not designed for swimming, showering or submersion. The charging case is not water resistant."),
    ("page",),
    ("h", "6. Error codes"),
    ("p", "Error codes are shown in the Lumora app and announced by a voice prompt in the earbud."),
    ("table", [
        ["Code", "Meaning", "What to do"],
        ["E-101", "Left earbud not detected in the case", "Clean the gold charging contacts on the earbud and inside the case with a dry cotton swab. Reinsert the earbud."],
        ["E-214", "Battery temperature sensor fault in the RIGHT earbud", "Let the earbuds rest at room temperature for 20 minutes outside the case. Update to firmware 3.2.1 or later. If the code appears again, contact support for a replacement right earbud."],
        ["E-305", "Charging case firmware does not match earbud firmware", "Open the Lumora app with the case open and both earbuds inside; the app will offer a case update."],
        ["E-410", "Pairing memory full (8 devices)", "Reset the earbuds: hold the case button for 15 seconds until the LED blinks amber 3 times."],
    ]),
    ("h", "7. Factory reset"),
    ("p", "Place both earbuds in the case, keep the lid open, and hold the case button for 15 seconds until the LED blinks amber 3 times. Then remove 'Lumora Pulse' from your phone's Bluetooth list before pairing again."),
    ("h", "8. Specifications"),
    ("table", [
        ["Item", "Value"],
        ["Driver", "10 mm dynamic"],
        ["Bluetooth", "5.3, SBC and AAC"],
        ["Weight", "4.8 g per earbud, 42 g case"],
        ["Multipoint", "Not supported"],
    ]),
]

PULSE_PRO_MANUAL = [
    ("h", "Lumora Pulse Pro - User Manual"),
    ("p", "Model: LA-PL-200. Document version 5.0, March 2026."),
    ("h", "1. Noise control modes"),
    ("p", "Pulse Pro has three modes: ANC (active noise cancellation), Transparency and Off. Press and hold either earbud for 2 seconds to cycle through the modes. Adaptive ANC adjusts the strength to your surroundings and can be enabled in the Lumora app."),
    ("h", "2. Battery"),
    ("p", "Up to 6 hours of playback with ANC on, or 8 hours with ANC off. The case adds up to 24 hours. The case supports wireless Qi charging and USB-C. A 5-minute charge gives 1 hour of playback."),
    ("h", "3. Water resistance"),
    ("p", "The earbuds are rated IP54 (dust and splash resistant). The case is rated IPX2. Do not submerge the earbuds."),
    ("page",),
    ("h", "4. Error codes"),
    ("p", "Note: Pulse Pro uses its own error code list. Codes are not interchangeable with the standard Pulse."),
    ("table", [
        ["Code", "Meaning", "What to do"],
        ["E-101", "Earbud not detected in the case", "Clean the charging contacts and reinsert the earbud."],
        ["E-214", "ANC microphone calibration failed", "Remove the ear tips, place both earbuds in the open case, and run 'Recalibrate ANC' in the Lumora app (Device > Noise control). Keep the room quiet during the 30-second calibration."],
        ["E-502", "ANC processor temperature too high", "Stop using ANC for 10 minutes and keep the earbuds out of direct sunlight."],
        ["E-610", "Wireless charging coil fault in the case", "Charge the case with USB-C. If wireless charging still fails after 24 hours, contact support."],
    ]),
    ("h", "5. Specifications"),
    ("table", [
        ["Item", "Value"],
        ["Driver", "11 mm dynamic with graphene diaphragm"],
        ["Bluetooth", "5.4, SBC, AAC and LDAC"],
        ["Microphones", "6 (3 per earbud)"],
        ["Multipoint", "Supported (2 devices)"],
    ]),
]

ARIA_X2_MANUAL = [
    ("h", "Lumora Aria X2 - User Manual"),
    ("p", "Model: LA-AX2. Document version 2.2, January 2026."),
    ("h", "1. Getting started"),
    ("p", "Press the power button for 2 seconds. The speaker enters pairing mode automatically the first time. Select 'Lumora Aria X2' in your device's Bluetooth menu."),
    ("h", "2. Party Mode"),
    ("p", "Party Mode links several Aria X2 speakers so they play the same music. Press the Party button on the first speaker, then on each additional speaker within 30 seconds. Up to 4 Aria X2 speakers can be linked in one Party Mode session. Party Mode works only between Aria X2 speakers."),
    ("h", "3. Battery"),
    ("p", "Up to 20 hours of playback at 50% volume, about 9 hours at maximum volume. Charging from empty to full takes 3.5 hours with a 20 W USB-C charger. The speaker can charge a phone through its USB-C port (power bank function)."),
    ("page",),
    ("h", "4. Water and dust resistance"),
    ("p", "Aria X2 is rated IP67: dust tight, and protected against immersion in fresh water up to 1 meter for 30 minutes. The rubber cover over the USB-C port must be fully closed for the rating to apply. After exposure to salt water or chlorinated water, rinse the speaker with fresh water and let it dry for 2 hours before charging."),
    ("h", "5. Specifications"),
    ("table", [
        ["Item", "Value"],
        ["Output power", "2 x 15 W"],
        ["Frequency response", "55 Hz - 20 kHz"],
        ["Bluetooth", "5.3, range up to 30 m"],
        ["Weight", "890 g"],
        ["Multipoint", "Supported (from firmware 2.2.0)"],
    ]),
]

HALO_ANC_MANUAL = [
    ("h", "Lumora Halo ANC - User Manual"),
    ("p", "Model: LA-HL-300. Document version 1.4, November 2025."),
    ("h", "1. Controls"),
    ("p", "The right ear cup has a touch surface. Swipe up or down to change the volume, swipe forward or back to skip tracks, and double tap to play or pause. The ANC button on the left ear cup switches between ANC, Transparency and Off."),
    ("h", "2. Battery"),
    ("p", "Up to 40 hours with ANC on and 55 hours with ANC off. A 10-minute charge gives 6 hours of playback. The headphones can be used with the included 3.5 mm cable when the battery is empty (ANC is not available in wired mode)."),
    ("h", "3. Wear detection"),
    ("p", "Music pauses when you take the headphones off and resumes when you put them back on. Wear detection can be switched off in the Lumora app."),
    ("h", "4. Care"),
    ("p", "Ear cushions are replaceable. Clean them with a slightly damp cloth; do not use alcohol. Halo ANC has no water resistance rating: keep it away from rain and sweat-heavy workouts."),
    ("h", "5. Specifications"),
    ("table", [
        ["Item", "Value"],
        ["Driver", "40 mm dynamic"],
        ["Bluetooth", "5.3, SBC, AAC and LDAC"],
        ["Weight", "265 g"],
        ["Multipoint", "Supported (2 devices)"],
    ]),
]

WARRANTY_TERMS = [
    ("h", "Lumora Limited Warranty - Terms and Conditions"),
    ("p", "Effective for products purchased on or after January 1, 2026."),
    ("h", "1. Warranty period"),
    ("p", "The warranty period depends on the product and on the country where it was purchased. The applicable period for each product and region is listed in the Lumora Warranty Matrix. The warranty starts on the date of original purchase and is not extended by a repair or replacement; a replacement product or part carries the remaining warranty of the original product."),
    ("h", "2. What is covered"),
    ("p", "Defects in materials and workmanship under normal use. Batteries are covered if their capacity drops below 70% of the original rated capacity within the warranty period."),
    ("h", "3. Water damage"),
    ("p", "Water damage is covered only for products with an IP rating and only within the limits of that rating. For Aria X2 (IP67), damage from immersion in fresh water up to 1 meter for up to 30 minutes is covered, provided the USB-C port cover was fully closed. Damage from salt water, chlorinated pool water, hot water or steam, or from immersion with the port cover open, is not covered. For Pulse (IPX4), Pulse Pro (IP54) and Aria Mini (IPX5), damage from sweat, rain and splashes is covered; damage from immersion is not covered. Halo ANC and Halo Lite have no IP rating and water damage is never covered."),
    ("page",),
    ("h", "4. What is not covered"),
    ("p", "Physical damage (drops, crushing, pets), loss or theft, normal wear of ear tips and ear cushions, cosmetic damage, damage from unauthorized repairs, and products whose serial number has been removed."),
    ("h", "5. How to make a claim"),
    ("p", "Contact Lumora support with the product serial number. For products not bought directly from Lumora, include a proof of purchase showing the purchase date. Support will first help with troubleshooting and then, if needed, arrange a replacement. Lumora does not repair units; defective units are replaced with new or equivalent-to-new products."),
    ("h", "6. Consumer rights"),
    ("p", "This warranty is in addition to, and does not limit, your statutory rights under the consumer law of your country."),
]

RESELLER_GUIDE = [
    ("h", "Lumora Authorized Reseller Program - Partner Guide 2026"),
    ("p", "For retail and online partners. Confidential."),
    ("h", "1. Joining the program"),
    ("p", "Resellers must operate a physical store or an online shop with its own domain, and must be registered as a business. The minimum opening order is USD 2,500 (or EUR/GBP equivalent). Applications are reviewed by the Channel Sales team within 10 business days."),
    ("h", "2. Partner tiers and margins"),
    ("table", [
        ["Tier", "Annual purchase volume", "Margin on MSRP", "Payment terms"],
        ["Standard", "Below USD 50,000", "30%", "Prepayment"],
        ["Gold", "USD 50,000 - 250,000", "35%", "Net 30"],
        ["Platinum", "Above USD 250,000", "38%", "Net 45"],
    ]),
    ("h", "3. Pricing rules"),
    ("p", "Resellers must respect the Minimum Advertised Price (MAP), which is 90% of the MSRP. Discounts below MAP may be offered in store but not advertised online. Repeated MAP violations lead to suspension from the program."),
    ("h", "4. Demo units"),
    ("p", "Partners can buy demo units at 50% of the dealer price, with a maximum of 2 demo units per product per store."),
    ("h", "5. Warranty handling"),
    ("p", "End customers of resellers contact Lumora support directly for warranty claims. Resellers do not process warranty replacements themselves."),
]

PDFS = {
    "manuals/pulse-user-manual.pdf": PULSE_MANUAL,
    "manuals/pulse-pro-user-manual.pdf": PULSE_PRO_MANUAL,
    "manuals/aria-x2-user-manual.pdf": ARIA_X2_MANUAL,
    "manuals/halo-anc-user-manual.pdf": HALO_ANC_MANUAL,
    "policies/warranty-terms-2026.pdf": WARRANTY_TERMS,
    "internal/reseller-program-guide-2026.pdf": RESELLER_GUIDE,
}


def build_pdf(path: Path, blocks: list[tuple]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title(blocks[0][1])
    pdf.set_author("Lumora Audio")
    pdf.add_page()
    for block in blocks:
        kind = block[0]
        if kind == "h":
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(3)
            pdf.multi_cell(0, 7, block[1], new_x="LMARGIN", new_y="NEXT")
        elif kind == "p":
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, block[1], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif kind == "table":
            pdf.set_font("Helvetica", "", 9)
            with pdf.table(headings_style=FontFace(emphasis="BOLD")) as table:
                for row in block[1]:
                    table.row(row)
            pdf.ln(2)
        elif kind == "page":
            pdf.add_page()
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


# ---------------------------------------------------------------------------
# Excel workbooks
# ---------------------------------------------------------------------------

CATALOG = [
    ["SKU", "Product", "Category", "Price USD", "Price EUR", "Price GBP", "Battery (h, per charge)",
     "Total with case (h)", "Water rating", "Launch", "Status"],
    ["LA-PL-100", "Pulse", "Earbuds", 99, 99, 89, 7, 28, "IPX4", "2025-03", "Active"],
    ["LA-PL-200", "Pulse Pro", "Earbuds", 179, 189, 169, 6, 30, "IP54", "2025-09", "Active"],
    ["LA-HL-300", "Halo ANC", "Headphones", 299, 319, 279, 40, None, "None", "2024-10", "Active"],
    ["LA-HL-150", "Halo Lite", "Headphones", 129, 139, 119, 50, None, "None", "2023-05", "Discontinued (March 2026)"],
    ["LA-AX2", "Aria X2", "Speaker", 149, 159, 139, 20, None, "IP67", "2025-06", "Active"],
    ["LA-AM1", "Aria Mini", "Speaker", 69, 79, 65, 12, None, "IPX5", "2024-04", "Active"],
]

COLORS = [
    ["SKU", "Color", "Available in"],
    ["LA-PL-100", "Graphite", "All regions"],
    ["LA-PL-100", "Sand", "All regions"],
    ["LA-PL-200", "Graphite", "All regions"],
    ["LA-PL-200", "Ocean Blue", "US, UK only"],
    ["LA-HL-300", "Graphite", "All regions"],
    ["LA-HL-300", "Cream", "EU, UK only"],
    ["LA-AX2", "Graphite", "All regions"],
    ["LA-AX2", "Forest Green", "All regions"],
    ["LA-AX2", "Coral", "US only"],
    ["LA-AM1", "Graphite", "All regions"],
    ["LA-AM1", "Sky", "All regions"],
]

# Warranty in months. "US region" = US, Canada, Australia, New Zealand.
# "EU region" = EU, UK, Norway, Switzerland.
WARRANTY = [
    ["SKU", "Product", "Warranty US region (months)", "Warranty EU region (months)",
     "Water damage covered", "Notes"],
    ["LA-PL-100", "Pulse", 12, 24, "Sweat and splashes only", ""],
    ["LA-PL-200", "Pulse Pro", 12, 24, "Sweat and splashes only", ""],
    ["LA-HL-300", "Halo ANC", 24, 24, "No", ""],
    ["LA-HL-150", "Halo Lite", 12, 24, "No", "Discontinued; supported until March 2028"],
    ["LA-AX2", "Aria X2", 24, 24, "Yes, fresh water immersion within IP67 limits",
     "Port cover must be closed; salt/chlorinated water excluded"],
    ["LA-AM1", "Aria Mini", 12, 24, "Sweat, rain and splashes only", ""],
]


def build_xlsx(path: Path, sheets: dict[str, list[list]]) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for column in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in column)
            ws.column_dimensions[column[0].column_letter].width = min(width + 2, 60)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

TICKET_FIELDS = ["ticket_id", "created_at", "channel", "product", "category", "country",
                 "subject", "customer_message", "resolution", "status"]

# fmt: off
TICKETS = [
    ("T-20417", "2026-07-02", "chat", "Pulse", "error-code", "DE", "E-214 after car trip", "My Pulse says error E-214 after I charged it in the car.", "Customer on fw 3.1.0. Explained heat threshold, let cool 20 min. Advised update when 3.2.1 released. Resolved.", "closed"),
    ("T-20433", "2026-07-03", "email", "Halo ANC", "hardware", "US", "Right cushion came off", "The right ear cushion keeps detaching.", "Serial HA24-xxxx. Sent cushion kit HA-CUSH-2 free, no RMA.", "closed"),
    ("T-20451", "2026-07-05", "chat", "Pulse Pro", "error-code", "UK", "E-214 on Pulse Pro", "Getting E-214 on my Pulse Pro, is my battery broken?", "Pulse Pro E-214 = ANC mic calibration. Walked through Recalibrate ANC in app. Resolved.", "closed"),
    ("T-20480", "2026-07-08", "email", "Pulse", "return", "NL", "Return opened earbuds", "I opened my Pulse 20 days ago, can I still return them?", "Delivered 2026-06-18, purchased before 2026-07-01 so old 30-day policy applies. Return approved.", "closed"),
    ("T-20502", "2026-07-10", "phone", "Aria X2", "warranty", "US", "Speaker died after lake", "My Aria X2 stopped working after it fell in a lake.", "Fresh water, port cover closed per customer, 1 m depth. Covered under IP67 terms. RMA DEF-WATER created.", "closed"),
    ("T-20519", "2026-07-12", "chat", "Aria X2", "warranty", "ES", "Pool accident", "Dropped my Aria X2 in the pool, now no sound.", "Chlorinated water not covered by warranty. Offered out-of-warranty replacement at 40% off.", "closed"),
    ("T-20544", "2026-07-15", "email", "Pulse", "return", "FR", "Return after 3 weeks", "I bought Pulse on July 10 and opened them. Can I return now (Aug 2)?", "Delivered 2026-07-12, new policy: opened earbuds 14 days. Window ended 2026-07-26. Return declined politely, offered troubleshooting.", "closed"),
    ("T-20566", "2026-07-18", "chat", "Halo Lite", "support", "US", "Still supported?", "Is Halo Lite still getting updates?", "Discontinued March 2026, firmware support and warranty until March 2028.", "closed"),
    ("T-20590", "2026-07-21", "email", "Pulse Pro", "refund", "US", "Refund for bulk order", "Please refund my order of 4 Pulse Pro ($716).", "Above $500: escalated with Request approval macro to Customer Operations Manager. Approved 2026-07-22.", "closed"),
    ("T-20611", "2026-07-24", "chat", "Pulse", "connectivity", "DE", "Left earbud drops on Pixel", "Left earbud disconnects in single mode on my Pixel with Android 15.", "Known issue. Disabled LE Audio in developer settings. Works. Fix planned fw 3.3.", "closed"),
    ("T-20637", "2026-07-27", "phone", "Aria Mini", "product", "UK", "Party mode with Aria X2", "Can I link my Aria Mini with my friend's Aria X2?", "Party Mode only between same model. Explained.", "closed"),
    ("T-20658", "2026-07-30", "email", "Pulse", "shipping", "AU", "Import charges", "Why did I pay import fees on my Pulse order?", "Australia orders may have local import charges paid by recipient per shipping policy. No refund of charges.", "closed"),
    ("T-20679", "2026-08-02", "chat", "Halo ANC", "safety", "US", "Headphones got very hot", "My Halo ANC got really hot while charging and smells weird.", "priority:high safety. Told to stop charging. Product Safety arranged hazmat pickup. Full refund issued.", "closed"),
    ("T-20702", "2026-08-05", "email", "Pulse Pro", "error-code", "IT", "E-502", "Error E-502 when using ANC on the beach.", "ANC processor overheating. Advised 10 min rest, avoid direct sun.", "closed"),
    ("T-20725", "2026-08-08", "chat", "Pulse", "account", "US", "Device registered to other account", "App says device not registered, bought second hand.", "Previous owner must remove device. Customer to contact seller. Pending proof of purchase.", "open"),
    ("T-20749", "2026-08-11", "email", "Aria X2", "product", "CA", "How many speakers in party mode", "How many Aria X2 can I connect together?", "Up to 8 since firmware 2.3.0 (manual still says 4). Advised update.", "closed"),
    ("T-20771", "2026-08-14", "chat", "Pulse", "warranty", "US", "Warranty length", "How long is the warranty on Pulse in the US?", "US region: 12 months.", "closed"),
    ("T-20790", "2026-08-16", "chat", "Pulse", "warranty", "DE", "Warranty length Germany", "How long is the Pulse warranty in Germany?", "EU region: 24 months.", "closed"),
    ("T-20812", "2026-08-19", "email", "Halo ANC", "refund", "UK", "Goodwill for late delivery", "My order arrived 9 days late, can I get something?", "Goodwill credit GBP 20 applied (within $40/12 months cap).", "closed"),
    ("T-20833", "2026-08-21", "phone", "Pulse Pro", "hardware", "US", "Lost one earbud", "I lost my right Pulse Pro earbud.", "Loss not covered. Sent link to buy spare right earbud.", "closed"),
    ("T-20856", "2026-08-24", "chat", "Aria X2", "product", "NO", "Charge phone from speaker", "Can Aria X2 charge my phone?", "Yes, power bank function via USB-C port.", "closed"),
    ("T-20878", "2026-08-27", "email", "Pulse", "error-code", "UK", "E-305 after update", "Earbuds show E-305 after the app update.", "Case firmware mismatch. Updated case with lid open. Resolved.", "closed"),
    ("T-20899", "2026-08-29", "chat", "Pulse", "error-code", "US", "E-214 again at home", "E-214 keeps appearing at room temperature, already on 3.2.1.", "Genuine sensor fault. RMA DEF-BATT part R created, advance replacement.", "closed"),
    ("T-20921", "2026-09-01", "email", "Pulse Pro", "return", "US", "Return opened Pulse Pro", "Opened my Pulse Pro 10 days ago, want to return.", "Within 14-day opened window. Return label sent.", "closed"),
    ("T-20944", "2026-09-03", "chat", "Aria Mini", "warranty", "US", "Aria Mini in the rain", "My Aria Mini got wet in the rain and crackles now.", "IPX5 covers rain. US warranty 12 months, purchased 2026-02. RMA DEF-WATER created.", "closed"),
    ("T-20967", "2026-09-05", "email", "Halo ANC", "b2b", "DE", "Reseller refund request", "We are a reseller and need a refund for 12 damaged units.", "B2B order: routed to Channel Sales Lead for approval.", "open"),
    ("T-20988", "2026-09-08", "chat", "Pulse", "product", "US", "Wireless charging", "Can I charge the Pulse case on my wireless charger?", "No, Pulse case is USB-C only. Pulse Pro supports Qi.", "closed"),
    ("T-21010", "2026-09-10", "phone", "Aria X2", "warranty", "US", "Salt water", "Took my Aria X2 to the beach, it went in the sea.", "Salt water excluded from warranty. Offered rinse instructions; unit did not recover. Out-of-warranty offer.", "closed"),
    ("T-21033", "2026-09-12", "chat", "Pulse Pro", "refund", "US", "Refund $320", "Requesting refund of $320 for two defective Pulse Pro.", "Between $150 and $500: approved by Senior Support Specialist.", "closed"),
    ("T-21055", "2026-09-15", "email", "Pulse", "shipping", "JP", "Shipping to Japan", "Do you ship to Japan?", "Not a supported shipping country. Order cancelled and refunded.", "closed"),
]
# fmt: on

PARTNER_FIELDS = ["partner", "type", "country", "city", "carries", "demo_units", "tier"]

# fmt: off
PARTNERS = [
    ("SoundHaus Berlin", "store", "DE", "Berlin", "Pulse; Pulse Pro; Halo ANC; Aria X2", "yes", "Gold"),
    ("Klangwerk Online", "online", "DE", "Munich", "All active products", "no", "Platinum"),
    ("AudioCorner", "store", "UK", "Manchester", "Pulse; Pulse Pro; Aria X2; Aria Mini", "yes", "Standard"),
    ("Hi-Fi Lane", "store", "UK", "London", "Halo ANC; Pulse Pro", "yes", "Gold"),
    ("Beat Street Audio", "store", "US", "Austin", "All active products", "yes", "Gold"),
    ("GadgetGrove", "online", "US", "Seattle", "Pulse; Aria Mini", "no", "Standard"),
    ("Nordic Sound", "store", "NO", "Oslo", "Pulse Pro; Halo ANC; Aria X2", "yes", "Standard"),
    ("Ecoute", "store", "FR", "Lyon", "Pulse; Pulse Pro; Aria X2", "yes", "Standard"),
    ("TuneBox", "online", "NL", "Rotterdam", "All active products", "no", "Gold"),
    ("OzAudio", "store", "AU", "Melbourne", "Pulse; Pulse Pro; Aria X2", "yes", "Standard"),
]
# fmt: on


def build_csv(path: Path, fields: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)


def main() -> None:
    for rel, blocks in PDFS.items():
        build_pdf(RAW / rel, blocks)
    build_xlsx(RAW / "catalog/product-catalog-2026.xlsx", {"Products": CATALOG, "Colors": COLORS})
    build_xlsx(RAW / "catalog/warranty-matrix.xlsx", {"Warranty": WARRANTY})
    build_csv(RAW / "support/tickets-2026-q3.csv", TICKET_FIELDS, TICKETS)
    build_csv(RAW / "support/retail-partners.csv", PARTNER_FIELDS, PARTNERS)

    files = sorted(p for p in RAW.rglob("*") if p.is_file())
    print(f"{len(files)} files in data/raw:")
    for p in files:
        print(f"  {p.relative_to(RAW).as_posix():45} {p.stat().st_size:>7,} bytes")


if __name__ == "__main__":
    main()
