"""Normalization of ABO seller-entered attributes into closed label sets.

Raw values are lowercased and stripped before lookup. Anything not in a map
becomes None and is left out of that attribute's evaluation, so every metric
reports its own denominator.
"""
from .abo import en_values, product_type

COLORS = [
    "black", "white", "grey", "brown", "beige", "blue", "red", "green", "pink",
    "purple", "yellow", "orange", "gold", "silver", "clear", "multicolor",
]

_COLOR_MAP = {
    "black": "black", "matte black": "black", "carbon black": "black",
    "white": "white", "off-white": "white", "soft white": "white", "bright white": "white",
    "grey": "grey", "gray": "grey",
    "brown": "brown", "bronze": "brown", "copper": "brown", "mocha": "brown", "terracotta": "brown",
    "beige": "beige", "ivory": "beige", "cream": "beige", "tan": "beige", "sand": "beige", "linen": "beige",
    "blue": "blue", "navy": "blue", "navy blue": "blue", "light blue": "blue", "sky blue": "blue",
    "turquoise": "blue",
    "red": "red", "burgundy": "red",
    "green": "green", "pine green": "green",
    "pink": "pink",
    "purple": "purple",
    "yellow": "yellow",
    "orange": "orange",
    "gold": "gold",
    "silver": "silver",
    "clear": "clear", "transparent": "clear",
    "multi": "multicolor", "multi-colored": "multicolor", "multicolour": "multicolor",
    "multicoloured": "multicolor", "multi-coloured": "multicolor", "multicolor": "multicolor",
}

MATERIALS = [
    "metal", "wood", "leather", "fabric", "plastic", "glass", "ceramic", "rubber", "paper",
]

# Deliberately unmapped: "stone" (mostly on rugs, sofas, chairs, so not a real
# material label), "synthetic" and "mesh" (almost only shoes, and not visually
# separable from fabric or leather), "polyurethane" (used for both PU leather
# and foam), "other", "not applicable", and compound values such as
# "velvet, hardwood frame, metal legs".
_MATERIAL_MAP = {
    **dict.fromkeys([
        "metal", "stainless steel", "steel", "aluminium", "aluminum", "brass", "iron",
        "alloy steel", "cast iron", "zinc alloy", "copper", "bronze", "pewter", "silver",
        "sterling-silver", "sterling silver", "precious metal", "gold", "titanium",
    ], "metal"),
    **dict.fromkeys([
        "wood", "engineered wood", "mdf", "pine", "hardwood", "rubber wood", "oak",
        "bamboo", "solid wood", "plywood", "teak", "acacia", "mango wood", "walnut",
    ], "wood"),
    **dict.fromkeys([
        "leather", "faux leather", "leatherette", "pu", "suede", "genuine leather",
        "vegan leather", "pu leather",
    ], "leather"),
    **dict.fromkeys([
        "fabric", "polyester", "cotton", "100% cotton", "100% polyester", "canvas", "wool",
        "linen", "velvet", "nylon", "microfiber", "microfibre", "textile", "satin",
        "flannel", "jute", "knit", "denim", "terrycloth", "polycotton", "egyptian-cotton",
        "spun_polyester", "melange cotton yarn", "chenille", "fleece", "silk",
    ], "fabric"),
    **dict.fromkeys([
        "plastic", "polypropylene", "pvc", "pvc vinyl", "abs", "acrylic", "polycarbonate",
        "polyresin", "resin", "vinyl", "melamine", "polyethylene",
    ], "plastic"),
    **dict.fromkeys(["glass", "tempered_glass", "tempered glass", "crystal"], "glass"),
    **dict.fromkeys(["ceramic", "stoneware", "porcelain", "earthenware"], "ceramic"),
    **dict.fromkeys(["rubber", "silicone", "neoprene", "foam", "eva"], "rubber"),
    **dict.fromkeys(["paper", "cardboard", "paperboard"], "paper"),
}


def normalize_color(raw: str | None) -> str | None:
    return _COLOR_MAP.get(raw.strip().lower()) if raw else None


def normalize_material(raw: str | None) -> str | None:
    return _MATERIAL_MAP.get(raw.strip().lower()) if raw else None


def extract_labels(listing: dict) -> dict:
    """First English standardized color and first English material, normalized."""
    colors = en_values(listing, "color", "standardized_values")
    materials = en_values(listing, "material")
    return {
        "product_type": product_type(listing),
        "color": normalize_color(colors[0]) if colors else None,
        "material": normalize_material(materials[0]) if materials else None,
        "color_raw": colors[0] if colors else None,
        "material_raw": materials[0] if materials else None,
    }
