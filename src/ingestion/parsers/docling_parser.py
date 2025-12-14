import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
import json
from typing import Tuple
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

class DoclingParser:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        pipe = PdfPipelineOptions(do_table_structure=True, do_ocr=False)
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe)}
        )

    def parse(self, pdf_path: Path) -> Tuple[object, Path]:
        res = self.converter.convert(str(pdf_path))
        print("pdf_path", pdf_path)
        ddoc = res.document
        json_path = self.artifacts_dir / f"{pdf_path.stem}.json"
        json_path.write_text(
            json.dumps(ddoc.export_to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        return ddoc, json_path