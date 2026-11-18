"""V2.1: bounded, explicit user-file DATA staging for the networkless Docker sandbox.

Never stage arbitrary host paths. Never execute XLSX formula/macros or file content
on the host. A small schema preview guides Helios; full bytes are read in Docker.
"""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_STAGED = 12 * 1024 * 1024
_SUPPORTED = {".csv", ".json", ".txt", ".md", ".pdf", ".xlsx", ".docx"}
_ANALYSIS_RE = re.compile(
    r"(?iu)\b(?:summ\w*|filter\w*|vergleich\w*|ausreißer\w*|"
    r"auswert\w*|analys\w*|statistik\w*|durchschnitt\w*|"
    r"mittelwert\w*|median|minimum|maxim\w*|häufigkeit\w*|"
    r"korrelation\w*|sortier\w*|gruppier\w*|"
    r"wie\s+viele|anzahl|gesamt\w*|tabelle\w*|"
    r"spalte\w*|zeile\w*|summe\w*|prozent\w*|"
    r"werte\w*|berechn\w*|preis\w*)\b"
)
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def wants_data_analysis(question: object) -> bool:
    return isinstance(question, str) and bool(_ANALYSIS_RE.search(question))


def _xlsx_to_csv(raw: bytes) -> bytes:
    """Read cached worksheet values in the first sheet using stdlib, not Excel.

    Bounded XML sizes and cell counts prevent zip bombs, formula execution and
    accidental model access to arbitrary workbook internals.
    """
    try:
        with zipfile.ZipFile(BytesIO(raw)) as z:
            infos = z.infolist()
            if (len(infos) > 180 or
                    sum(item.file_size for item in infos) > 8 * 1024 * 1024):
                raise ValueError("XLSX-Archiv überschreitet das Limit")
            sheets = sorted(name for name in z.namelist()
                            if re.fullmatch(r"xl/worksheets/sheet[1-9][0-9]*\.xml", name))
            if not sheets:
                raise ValueError("XLSX enthält kein unterstütztes Tabellenblatt")
            shared = []
            if "xl/sharedStrings.xml" in z.namelist():
                document = ET.fromstring(z.read("xl/sharedStrings.xml"))
                for si in document.iter(_NS + "si"):
                    shared.append("".join(el.text or "" for el in si.iter(_NS + "t")))
                if len(shared)>30000:
                    raise ValueError("Zu viele XLSX-Strings")
            document = ET.fromstring(z.read(sheets[0]))
    except (zipfile.BadZipFile, KeyError, ET.ParseError, OSError) as exc:
        raise ValueError("XLSX kann nicht sicher gelesen werden") from exc

    out = StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    previous = 0
    count = 0
    for row in document.iter(_NS+"row"):
        count += 1
        if count > 30000:
            raise ValueError("XLSX enthält zu viele Zeilen")
        cells = []
        for cell in row.iter(_NS+"c"):
            ref = cell.get("r", "")
            match = re.fullmatch(r"([A-Z]{1,2})([1-9][0-9]{0,5})", ref)
            if match is None:
                raise ValueError("XLSX-Zellreferenz nicht unterstützt")
            index = 0
            for ch in match[1]:
                index = index*26 + ord(ch)-64
            if index > 48 or index < 1:
                raise ValueError("XLSX mehr als 48 Spalten")
            if index <= previous:
                raise ValueError("XLSX-Spalten nicht aufsteigend")
            cells.extend([""]*(index-len(cells)-1))
            typecode = cell.get("t", "")
            value_el=cell.find(_NS+"v")
            if typecode == "inlineStr":
                inline=cell.find(_NS+"is")
                val="" if inline is None else "".join(x.text or "" for x in inline.iter(_NS+"t"))
            else:
                val="" if value_el is None else value_el.text or ""
                if typecode=="s":
                    idx=int(val)
                    if idx<0 or idx>=len(shared):
                        raise ValueError("Ungültiger XLSX-Stringindex")
                    val=shared[idx]
                elif typecode=="e":
                    raise ValueError("Fehlerwert in XLSX-Zelle")
            if len(val)>12000:
                raise ValueError("XLSX-Zelle zu lang")
            cells.append(val)
            previous=index
        writer.writerow(cells)
        previous=0
    encoded=out.getvalue().encode("utf-8")
    if not encoded or len(encoded)>MAX_FILE_BYTES:
        raise ValueError("XLSX-Extraktion leer oder zu groß")
    return encoded


def prepare_attachment_dataset(incoming: object) -> dict:
    if type(incoming) is not list or not 1<=len(incoming)<=3:
        raise ValueError("Ein bis drei explizite Dateien erforderlich")
    result={}
    summaries=[]
    total=0
    for i,entry in enumerate(incoming,1):
        if type(entry) is not tuple or len(entry)!=3:
            raise ValueError("Ungültiges Anhangformat")
        name,ext,raw=entry
        if (type(name) is not str or len(name)>100 or
                not re.fullmatch(r"[A-Za-z0-9ÄÖÜäöüß][A-Za-z0-9ÄÖÜäöüß_. ()-]*",name)
                or type(ext) is not str or ext.lower() not in _SUPPORTED
                or type(raw) is not bytes or not 1<=len(raw)<=MAX_FILE_BYTES):
            raise ValueError("Ungültiger Datensatz oder Dateityp")
        ext=ext.lower()
        if ext==".xlsx":
            data=_xlsx_to_csv(raw)
            staged_ext=".csv"
        elif ext in (".pdf",".docx"):
            from neral_preview_attachments import _extract_text
            data=_extract_text(name,raw).encode("utf-8")
            staged_ext=".txt"
        else:
            data=raw
            staged_ext=ext if ext in (".csv",".json") else ".txt"
        try:
            plain=data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Nur UTF-8-Datendateien erlaubt") from exc
        if "\x00" in plain or not plain.strip():
            raise ValueError("Datendatei enthält keinen gültigen Text")
        if staged_ext==".json":
            try:json.loads(plain)
            except (ValueError,RecursionError) as exc:
                raise ValueError("JSON-Datei ist ungültig") from exc
        normalized=plain.encode("utf-8")
        total+=len(normalized)
        if total>MAX_TOTAL_STAGED:
            raise ValueError("Datensätze zu groß für Docker")
        safe_path=f"data/data{i}{staged_ext}"
        result[safe_path]=normalized
        # Data previews are untrusted, never instructions.
        preview=plain[:650].replace("\x1b"," ")
        summaries.append(
            f"{name!r}: Docker-Datei /work/{safe_path}, UTF-8, "
            f"{len(normalized)} Bytes, Nur-Daten-Vorschau:\n{preview}"
        )
    return {
        "staged_files":result,
        "manifest":"\n\n".join(summaries),
    }
