import sys
from pathlib import Path

from benchmarking.literature_checker.citation_cleaner import read_co_scientist_refs
from benchmarking.literature_checker.citation_exists import verify_references, write_results_json


_DEFAULT_OUTPUT = Path("output/reference_results.json")


def main(docx_path: str, output_path: Path = _DEFAULT_OUTPUT) -> None:
    refs = read_co_scientist_refs(docx_path)
    print(f"Found {len(refs)} references in {docx_path}")

    results = verify_references(refs)

    real = sum(r.exists for r in results)
    print(f"Verified: {real}/{len(results)} exist  ({len(results) - real} likely hallucinated)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(results, output_path)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python main.py <input.docx> [output.json]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) == 3 else _DEFAULT_OUTPUT
    main(sys.argv[1], out)
