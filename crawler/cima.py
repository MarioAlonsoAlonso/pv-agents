"""Brand names of each active ingredient, from CIMA (the AEMPS medicines database).

People write "llevo un mes con Besitran" and never say "sertralina". The brand list is
official data, so it comes from CIMA's public REST API and not from the model (which
could invent brands). It is cached in ``crawler/data/marcas.json`` (versioned: it is public
data and makes the crawler reproducible); ``--refresh`` downloads it again.

    uv run python -m crawler.cima --refresh
"""

from __future__ import annotations

import argparse
import json
import re
import time

from crawler.config import BRANDS_FILE, NOISE_TERMS, WATCHLIST
from crawler.text import fold

CIMA_API = "https://cima.aemps.es/cima/rest"
# First words of a product name that are not a brand: generics are named after the active
# ingredient, and some names start with a form or a laboratory prefix.
_NOT_BRAND = {"acido", "sal", "sales"}
# Brands that are ordinary words: searching them brings only noise ("captor" = kidnapper).
_COMMON_WORDS = {"captor", "essential"}


def brand_of(product_name: str, ingredient: str, ingredients: str = "") -> str | None:
    """ "BESITRAN 50 mg comprimidos" -> "Besitran"; "SERTRALINA CINFA 50 mg" -> None.

    ``ingredients`` is CIMA's ``vtm`` ("dapagliflozina + metformina"): a name that starts
    with any of its ingredients is a generic of a combination, not a brand.
    """
    first = re.split(r"[\s/,]+", product_name.strip())[0]
    word = fold(first)
    if not word.isalpha() or len(word) < 4 or word in _NOT_BRAND | _COMMON_WORDS:
        return None
    if word in {fold(ingredient).split()[0], *re.findall(r"\w+", fold(ingredients))}:
        return None  # a generic
    return first.capitalize()


def fetch_brands(ingredient: str, client=None, pause: float = 0.3) -> list[str]:
    """All brands in CIMA whose first active ingredient is ``ingredient``."""
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=30)
    brands: set[str] = set()
    page = 1
    try:
        while True:
            response = client.get(
                f"{CIMA_API}/medicamentos", params={"practiv1": ingredient, "pagina": page}
            )
            response.raise_for_status()
            data = response.json()
            for product in data.get("resultados", []):
                if product.get("generico"):
                    continue  # EFG generics with a fancy name ("Essential") are not brands
                brand = brand_of(
                    product.get("nombre", ""),
                    ingredient,
                    (product.get("vtm") or {}).get("nombre", ""),
                )
                if brand:
                    brands.add(brand)
            total, size = data.get("totalFilas", 0), data.get("tamanioPagina", 25) or 25
            if page * size >= total:
                break
            page += 1
            time.sleep(pause)
    finally:
        if own:
            client.close()
    return sorted(brands)


def load_brands(path=BRANDS_FILE) -> dict[str, list[str]]:
    """``{ingredient: [brands]}`` from the cache (empty lists if it does not exist)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {drug: [] for drug in WATCHLIST}


def main(argv: list[str] | None = None) -> None:
    from crawler.text import utf8_stdout

    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="download the brands from CIMA")
    args = parser.parse_args(argv)
    if not args.refresh and BRANDS_FILE.exists():
        print(json.dumps(load_brands(), ensure_ascii=False, indent=2))
        return
    brands = {}
    for drug in (*WATCHLIST, *NOISE_TERMS):
        brands[drug] = fetch_brands(drug)
        print(f"{drug}: {', '.join(brands[drug]) or '(sin marcas)'}")
    BRANDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BRANDS_FILE.write_text(json.dumps(brands, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"guardado en {BRANDS_FILE}")


if __name__ == "__main__":
    main()
