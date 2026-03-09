# preprocessing.py
import pandas as pd
import numpy as np
import re


def load_fda_data(filepath: str) -> pd.DataFrame:
    """Load the FDA Dissolution Methods Database."""
    df = pd.read_csv(filepath)
    return df



# 1. Function for Cleaning Dosage Form Column

# Known "Dosage Form" release types to help detect them outside parentheses
KNOWN_RELEASE_TYPES = [
    "Extended Release", "Delayed Release", "Orally Disintegrating", 
    "Immediate Release", "Controlled Release", "Sustained Release", 
    "Copackage" # , "For Suspension"
]

def clean_dosage_form_data(value):
    """
    Separates raw "Dosage Form" data entries into "Dosage Form" and "Release Type".
    """
    if pd.isna(value):
        return ("Unknown", "Unknown")
    
    # Strip spaces
    val = str(value).strip()

    # # Look for release type in parentheses
    # m = re.match(r"^(.*?)\s*\((.*?)\)$", val)
    
    # Regex: capture base form + anything inside parentheses
    # e.g. "Tablet (Delayed Release, Orally Disintegrating)"
    m = re.match(r"^(.*?)\s*(?:\((.+)\))?$", val)
    # m = re.match(r"^([A-Za-z ]+?)(?:\s*\((.+)\))?$", val)
    if m:
        dosage_form = m.group(1).strip().title()  # normalize dosage form
        release_type = m.group(2).strip() if m.group(2) else None

        # Remove trailing commas from both
        dosage_form = dosage_form.rstrip(',').strip()
        dosage_form = re.sub(r"\s+", " ", dosage_form)
        if release_type:
            release_type = release_type.rstrip(',').strip() # Note: ODT refers to Orally Disintegrating Tablet
            release_type = re.sub(r"\s+", " ", release_type)
            if release_type and not release_type[0].isupper():
                release_type = release_type[0].upper() + release_type[1:]

        # If no parentheses captured and there’s a comma, check for trailing release type
        if not release_type and ',' in dosage_form:
            parts = [p.strip() for p in dosage_form.split(',')]
            # If last part looks like a known release type, separate it
            if parts[-1] in KNOWN_RELEASE_TYPES:
                release_type = parts[-1]
                dosage_form = ', '.join(parts[:-1]).rstrip(',').strip()
        
        return dosage_form, release_type
    else:
        # fallback: keep whole thing in base, no modifiers
        return val.rstrip(',').title(), None



# 2. Function for Cleaning Apparatus Column

def clean_apparatus_column(df, col='USP Apparatus', cleaned_col_name='Apparatus_Cleaned'):
    """
    Simplify the USP Apparatus column into clean standardized categories.
    """
    apparatus_map = {
        'I': 'Basket',
        'II': 'Paddle',
        'III': 'Reciprocating Cylinder',
        'IV': 'Flow-through Cell',
        'V': 'Paddle over Disk',
        'VI': 'Cylinder',
        'VII': 'Reciprocating Holder'
    }

    def extract_apparatus(text):
        if not isinstance(text, str) or text.strip() == "":
            return np.nan

        text_clean = re.sub(r'\s+', ' ', text.strip())  # normalize spacing
        text_lower = text_clean.lower()

        # Capture Roman numeral at the start
        match = re.match(r'\b([IVX]+)\b', text_clean)
        if match:
            code = match.group(1)
            if code in apparatus_map:
                return apparatus_map[code]

        # Fallbacks (strict ordering, specific → general)
        if "reciprocating cylinder" in text_lower:
            return "Reciprocating Cylinder"
        elif "reciprocating holder" in text_lower or "holder" in text_lower:
            return "Reciprocating Holder"
        elif "paddle over disk" in text_lower or re.search(r'\bpaddle\s+over\s+disk\b', text_lower):
            return "Paddle over Disk"
        elif "flow-through" in text_lower or "flow through" in text_lower:
            return "Flow-through Cell"
        elif re.search(r'\bbasket\b', text_lower):
            return "Basket"
        elif re.search(r'\bpaddle\b', text_lower):
            return "Paddle"
        elif re.search(r'\bcylinder\b', text_lower):
            return "Cylinder"
        
        # # Fallbacks
        # if "paddle over disk" in text_lower or "disk" in text_lower:
        #     return "Paddle over Disk"
        # elif "flow" in text_lower:
        #     return "Flow-through Cell"
        # elif "reciprocating" in text_lower and "cylinder" in text_lower:
        #     return "Reciprocating Cylinder"
        # elif "reciprocating" in text_lower:
        #     return "Reciprocating Holder"
        # elif "basket" in text_lower:
        #     return "Basket"
        # elif "cylinder" in text_lower:
        #     return "Cylinder"
        # elif "paddle" in text_lower:
        #     return "Paddle"
        # else:
        #     return np.nan  # Non-standard entry

    df[cleaned_col_name] = df[col].apply(extract_apparatus)
    return df



# 3. Functions for Cleaning Speed (RPM) Column

# Apparatus → valid agitation unit keywords
APPARATUS_UNITS = {
    "Paddle": ["rpm"],
    "Basket": ["rpm"],
    "Paddle over Disk": ["rpm"],
    "Cylinder": ["dpm", "cycles", "cpm"],
    "Reciprocating Cylinder": ["dpm", "cycles", "cpm"],
    "Reciprocating Holder": ["dpm", "cycles", "cpm"],
    "Flow-through Cell": [],  # flow rate, not agitation
}

# Step A of simplifing the Speed (RPM) column into clean numerical and standardized values
def clean_speed_text(text):
    """
    Part of simplifing the Speed (RPM) column into clean numerical and standardized values.
    
    Step A: Cleaning the text:
    Keep only strings that contain numbers and valid units (rpm, dpm, cpm, cycles, dips).
    Ignore irrelevant strings (chews, flow, ml/min).
    """
    
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    if any(x in text_lower for x in ["flow", "ml/min", "chew", "orbit", "stroke"]):
        return None
    return text_lower


# Step B of simplifing the Speed (RPM) column into clean numerical and standardized values
def split_agitation_speeds_row(row, text_col='Speed (RPMs)', apparatus_col='Apparatus_Cleaned'):
    """
    Returns list of rows (dicts), one per valid agitation speed.
    Handles:
      - numbers with units (rpm, dpm, etc.)
      - plain numbers
      - missing apparatus (kept as None)
    """
    text = row[text_col]
    apparatus = row.get(apparatus_col, None)

    # Skip if text is missing (None or NaN)
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    
    # Convert everything else to string
    text_lower = str(text).lower()

    # Keep apparatus as None if missing
    if isinstance(apparatus, str):
        apparatus_clean = apparatus.strip()
        valid_units = APPARATUS_UNITS.get(apparatus_clean, [])
    else:
        apparatus_clean = None
        # If apparatus is missing, assume rpm; just keep numbers
        valid_units = []

    # Ignore irrelevant text
    if any(x in text_lower for x in ["flow", "ml/min", "stroke", "orbit", "chew"]):
        return []

    # 1. Extract numbers with units
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(rpm|dpm|cpm|cycles|dips)', text_lower)
    filtered_numbers = [float(val) for val, unit in matches if unit in valid_units]

    # 2. Fallback: if no matches, extract any plain numbers
    if not filtered_numbers:
        plain_numbers = re.findall(r'\d+', text_lower)
        if plain_numbers:
            filtered_numbers = [float(n) for n in plain_numbers]

    # 3. Duplicate row for each valid speed
    new_rows = []
    for speed in filtered_numbers:
        new_row = row.copy()
        new_row['Agitation_Speed'] = speed
        # Keep apparatus as None if missing
        new_row[apparatus_col] = apparatus_clean
        new_rows.append(new_row)

    return new_rows


    apparatus = apparatus.strip()
    text_lower = text.lower()

    # Ignore irrelevant text
    if any(x in text_lower for x in ["flow", "ml/min", "stroke", "orbit", "chew"]):
        return []

    # 1 Extract all numbers with valid units
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(rpm|dpm|cpm|cycles|dips)', text_lower)
    valid_units = APPARATUS_UNITS.get(apparatus, [])
    filtered_numbers = [float(val) for val, unit in matches if unit in valid_units]

    # 2 Fallback: if no matches, extract any plain numbers
    if not filtered_numbers:
        plain_numbers = re.findall(r'\d+', text_lower)
        # Only use as fallback if apparatus expects a numeric speed
        if valid_units and plain_numbers:
            filtered_numbers = [float(n) for n in plain_numbers]

    # 3 Duplicate row for each valid speed
    new_rows = []
    for speed in filtered_numbers:
        new_row = row.copy()
        new_row['Agitation_Speed'] = speed
        new_rows.append(new_row)

    return new_rows












# ============================================================
# 1 CONFIGURATION SECTION
# ============================================================

# Apparatus types that are rotational (i.e., expect RPM)
ROTATIONAL_APPARATUS = {
    "Paddle",
    "Basket",
    "Paddle over Disk"
}

# Apparatus types that are reciprocating (expect DPM/CPM/Cycles)
RECIPROCATING_APPARATUS = {
    "Cylinder",
    "Reciprocating Cylinder",
    "Reciprocating Holder"
}

# Words that indicate the entry is NOT agitation speed
IGNORED_KEYWORDS = [
    "flow",
    "ml/min",
    "stroke depth",
    "chew",
    "orbit"
]

# ============================================================
# 2 UNIT NORMALIZATION REGEX
# ============================================================

# This regex captures:
#  - 50 rpm
#  - 50 r/min
#  - 30 dpm
#  - 30 cycles per min
#  - 30 cycles/min
#  - 30 dips/min
#
# The regex is verbose for readability.
UNIT_REGEX = re.compile(r"""
    (\d+(?:\.\d+)?)              # Capture number (integer or decimal)
    \s*
    (
        rpm|
        r\/?min|
        dpm|
        cpm|
        dips?\/?min|
        cycles?\s*(?:per|\/)?\s*min
    )
""", re.IGNORECASE | re.VERBOSE)

# ============================================================
# 3 AGITATION NUMBER EXTRACTION FUNCTION
# ============================================================

def extract_agitation_numbers(text, apparatus=None):
    """
    Extracts valid agitation speeds from raw text.

    Rules:
    1. If number has a recognized unit → accept it.
    2. If no unit but exactly one number exists → assume RPM
       ONLY for rotational apparatus (or missing apparatus).
    3. Reject ambiguous multi-number entries without units.
    """

    # -------------------------
    # Validate type
    # -------------------------
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []

    if not isinstance(text, str):
        return []

    text_lower = text.lower()

    # -------------------------
    # Ignore obvious non-agitation contexts
    # -------------------------
    if any(keyword in text_lower for keyword in IGNORED_KEYWORDS):
        return []

    # -------------------------
    # 1 Numbers WITH units
    # -------------------------
    matches = UNIT_REGEX.findall(text_lower)

    if matches:
        # Always trust explicitly stated units
        return [float(value) for value, _ in matches]

    # -------------------------
    # 2 Plain-number fallback (assume RPM)
    # -------------------------
    numbers = re.findall(r'\d+(?:\.\d+)?', text_lower)

    if len(numbers) == 1:
        # Only assume RPM for rotational apparatus
        if apparatus in ROTATIONAL_APPARATUS or apparatus is None:
            return [float(numbers[0])]

    # Otherwise reject
    return []


# ============================================================
# 4 SCRAP CLASSIFICATION FUNCTION
# ============================================================

def determine_scrap_reason(text):
    """
    Classifies why a row was rejected.
    Useful for auditing and improving cleaning rules.
    """

    if text is None or (isinstance(text, float) and np.isnan(text)):
        return "missing"

    if not isinstance(text, str):
        return "non_string"

    text_lower = text.lower()

    if any(x in text_lower for x in ["flow", "ml/min"]):
        return "flow_rate_not_agitation"

    if any(x in text_lower for x in ["chew", "stroke", "orbit"]):
        return "motion_description"

    if not re.search(r'\d', text_lower):
        return "no_numbers"

    if not UNIT_REGEX.search(text_lower):
        return "numbers_without_units_or_ambiguous"

    return "unrecognized_format"


# ============================================================
# 5 ROW EXPLOSION FUNCTION
# ============================================================

def split_agitation_speeds_row_with_scrapping(row,
                               text_col='Speed (RPMs)',
                               apparatus_col='Apparatus_Cleaned'):
    """
    Processes one row and returns:

    - List of valid expanded rows
    - List of rejected rows (with scrap reason)
    """

    text = row.get(text_col)
    apparatus = row.get(apparatus_col)

    speeds = extract_agitation_numbers(text, apparatus)

    # -------------------------
    # VALID CASE
    # -------------------------
    if speeds:
        valid_rows = []

        for speed in speeds:
            new_row = row.copy()

            new_row["Agitation_Speed"] = speed

            # Flag whether RPM was assumed
            if isinstance(text, str) and not UNIT_REGEX.search(text):
                new_row["Speed_Assumed_Unit"] = "rpm_assumed"
            else:
                new_row["Speed_Assumed_Unit"] = "explicit_unit"

            valid_rows.append(new_row)

        return valid_rows, []

    # -------------------------
    # SCRAPPED CASE
    # -------------------------
    scrap_row = row.copy()
    scrap_row["Scrap_Reason"] = determine_scrap_reason(text)

    return [], [scrap_row]
    

def split_agitation_speeds_row(row,
                               text_col='Speed (RPMs)',
                               apparatus_col='Apparatus_Cleaned'):

    text = row.get(text_col)
    apparatus = row.get(apparatus_col)

    speeds = extract_agitation_numbers(text, apparatus)

    new_rows = []

    # -------------------------------------------------
    # 1 VALID OR ASSUMED SPEEDS
    # -------------------------------------------------
    if speeds:
        for speed in speeds:
            new_row = row.copy()
            new_row["Agitation_Speed"] = speed

            # Determine if unit was explicit
            if isinstance(text, str) and UNIT_REGEX.search(text):
                new_row["Speed_Status"] = "valid"
            else:
                new_row["Speed_Status"] = "assumed"

            new_row["Scrap_Reason"] = None

            new_rows.append(new_row)

        return new_rows, []

    # -------------------------------------------------
    # 2 MISSING SPEED
    # -------------------------------------------------
    if text is None or (isinstance(text, float) and np.isnan(text)):
        new_row = row.copy()
        new_row["Agitation_Speed"] = np.nan
        new_row["Speed_Status"] = "missing"
        new_row["Scrap_Reason"] = None
        return [new_row], []

    # -------------------------------------------------
    # 3 INVALID SPEED (keep row, flag it)
    # -------------------------------------------------
    new_row = row.copy()
    new_row["Agitation_Speed"] = np.nan
    new_row["Speed_Status"] = "invalid"
    new_row["Scrap_Reason"] = determine_scrap_reason(text)

    return [new_row], []


# ============================================================
# 6 APPLY TO ENTIRE DATAFRAME
# ============================================================

def clean_speed_column(df,
                       split_func,
                       text_col='Speed (RPMs)',
                       apparatus_col='Apparatus_Cleaned'):
    """
    Main pipeline function.

    Returns:
        clean_df      → expanded valid rows
        scrapped_df   → rejected rows with reasons
    """

    valid_rows = []
    scrapped_rows = []

    for _, row in df.iterrows():
        good, bad = split_func(
            row,
            text_col=text_col,
            apparatus_col=apparatus_col
        )

        valid_rows.extend(good)
        scrapped_rows.extend(bad)

    clean_df = pd.DataFrame(valid_rows)
    scrapped_df = pd.DataFrame(scrapped_rows)

    return clean_df, scrapped_df











# 4. Function for Medium Type Column

def extract_medium_type(text):
    """
    Extract and standardize the medium type from a messy text entry in the FDA Dissolution dataset.
    Returns:
        A list of standardized medium type strings.
        Returns [np.nan] if no valid medium type is found.
    """
    # Handle missing or non-string inputs
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return [np.nan]
    if not isinstance(text, str):
        return [np.nan]

    # Normalize text
    clean_text = text.strip().lower()

    # Ignore clearly irrelevant text
    if re.fullmatch(r"(develop a (dissolution|release) method|refer to (usp|fda's dissolution guidance)|n/?a|not applicable|none)", clean_text):
        return [np.nan]

    # Define regex patterns for common medium types
    patterns = {
        "hcl": r"(0\.\d+|1)\s*n?\s*hcl|simulated gastric fluid|s(g|i)f|dilute hcl",
        "phosphate_buffer": r"phosphate buffer|sodium phosphate|potassium phosphate|pbs",
        "acetate_buffer": r"acetate buffer|sodium acetate|acetic acid",
        "water": r"water|distilled water|h2o",
        "citrate_buffer": r"citrate buffer|sodium citrate",
        "sif": r"simulated intestinal fluid",
        "sgs": r"simulated gastric saliva",
        "surfactant": r"(sodium lauryl sulfate|sds|polysorbate|tween|ctab|sls)",
        "multi_stage": r"(acid stage|buffer stage|stage \d+|media \d+|phase \d+)",
    }

    # Check for each pattern
    medium_types = []
    for medium, pattern in patterns.items():
        if re.search(pattern, clean_text):
            medium_types.append(medium)

    # Special handling for multi-stage media
    if "multi_stage" in medium_types:
        # Extract acid and buffer stages
        acid_stage = re.search(r"acid stage:\s*(.*?)(?=buffer stage|$)", clean_text, re.IGNORECASE)
        buffer_stage = re.search(r"buffer stage:\s*(.*?)(?=acid stage|$)", clean_text, re.IGNORECASE)
        if acid_stage:
            medium_types.append(f"acid_stage_{acid_stage.group(1).strip()}")
        if buffer_stage:
            medium_types.append(f"buffer_stage_{buffer_stage.group(1).strip()}")

    # If no pattern matched, return np.nan
    if not medium_types:
        return [np.nan]

    # Remove duplicates while preserving order
    seen = set()
    unique_medium_types = [mt for mt in medium_types if not (mt in seen or seen.add(mt))]

    return unique_medium_types








INVALID_PATTERNS = [
    r"refer to",
    r"develop",
    r"guidance",
    r"characterize",
    r"method"
]

PH_REGEX = r"ph\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)"
# PH_REGEX = r"ph\s*([0-9.]+)"

SURFACTANTS = ["sds", "sls", "tween", "polysorbate"]

SURFACTANT_MAP = {
    "sds": "sodium dodecyl sulfate",
    "sls": "sodium lauryl sulfate",
    "sodium dodecyl sulfate": "sodium dodecyl sulfate",
    "sodium lauryl sulfate": "sodium lauryl sulfate",
    "tween": "tween",
    "polysorbate": "polysorbate",
    "ctab": "ctab",
    "cetyl": "ctab"
}

# Defining standard medium classes
MEDIUM_CLASSES = [
    "hcl",
    "sgf",
    "phosphate_buffer",
    "acetate_buffer",
    "citrate_buffer",
    "water",
    "surfactant_solution",
    "other"
]

MULTISTAGE_PATTERNS = [
    "stage",
    "tier",
    "pre-exposed",
    "then",
    "followed by"
]


# Defining function for removing non media entries
def remove_non_media(text):
    if pd.isna(text):
        return None
    text_lower = text.lower()
    if any(p in text_lower for p in INVALID_PATTERNS):
        return None
    return text


# Defining function for normalizing equivalent media
def normalize_medium(text):
    if pd.isna(text):
        return None
    text = text.lower()
    if "hcl" in text:
        return "hcl"
    if "sgf" in text or "simulated gastric" in text:
        return "sgf"
    if "phosphate" in text:
        return "phosphate_buffer"
    if "acetate" in text:
        return "acetate_buffer"
    if "water" in text:
        return "water"
    return "other"


# Defining holistic function for preprocessing medium types
def classify_medium(text):
    """
    Holistic function for preprocessing medium types.
    - Removes non media entries
    - Normalizes equivalent media
    - Extractes useful chemical features
    - Identifies surfactants
    - Classifies medium types into 8 generalized standard medium classes
    """
    if pd.isna(text):
        return None
    text = text.lower()
    # remove instruction-type rows
    if any(x in text for x in ["refer to", "develop", "guidance", "method"]):
        return None
    # main medium types
    if "sgf" in text or "simulated gastric" in text:
        return "sgf"
    if "hcl" in text or "hydrochloric" in text:
        return "hcl"
    if "phosphate" in text:
        return "phosphate_buffer"
    if "acetate" in text:
        return "acetate_buffer"
    if "citrate" in text:
        return "citrate_buffer"
    if "water" in text or "deionized" in text:
        return "water"
    # surfactants
    if any(x in text for x in ["sds", "sls", "tween", "polysorbate"]):
        return "surfactant_solution"
    return "other"


# Defining function for extracting useful chemical features
def extract_ph(text):
    if pd.isna(text):
        return None
    match = re.search(PH_REGEX, text.lower())
    if match:
        return float(match.group(1))
    return None


# Defining function for detecting surfactant
def detect_surfactant(text):
    if pd.isna(text):
        return 0
    text = text.lower()
    return int(any(s in text for s in SURFACTANTS))


# Defining refined function for detecting surfactant
def detect_surfactant_refined(text):
    if pd.isna(text):
        return pd.Series({
            "surfactant_presence": "no surfactant present",
            "surfactant_type": None
        })
    text = text.lower()
    for key, value in SURFACTANT_MAP.items():
        if key in text:
            return pd.Series({
                "surfactant_presence": "surfactant present",
                "surfactant_type": value
            })
    return pd.Series({
        "surfactant_presence": "no surfactant present",
        "surfactant_type": None
    })


# Defining function for detecting surfactant
def detect_multistage(text):
    if pd.isna(text):
        return 0
    return int("stage" in text.lower())


# Defining refined function for detecting surfactant
def detect_multistage_refined(text):
    if pd.isna(text):
        return 0
    text = text.lower()
    return int(any(p in text for p in MULTISTAGE_PATTERNS))





    







# 5. Function for Medium Volume Column

def extract_medium_volumes(text):
    """
    Extract medium volumes (in mL) from a messy text entry in the FDA Dissolution dataset.

    Returns:
        A list of numeric medium volumes (as ints).
        Returns an empty list if no valid numeric volumes are found.
    """
    
    # Handle missing or non-string inputs
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return [np.nan]

    # If it's a pure number (e.g., 900 or 900.0)
    if isinstance(text, (int, float)):
        return [int(text)]

    if not isinstance(text, str):
        return [np.nan]

    # Normalize text
    clean_text = text.strip().lower()

    # Ignore clearly irrelevant text
    if re.fullmatch(r"(use open mode|n/?a|not applicable|none)", clean_text):
        return []

    # Replace commas and strange delimiters with semicolons for uniform splitting
    clean_text = re.sub(r"[,/]", ";", clean_text)

    # Extract all numeric values followed by optional mL/ml or context
    matches = re.findall(r"(\d{2,5})\s*(?:m?l)?", clean_text)

    # Convert to integers safely
    volumes = []
    for m in matches:
        try:
            val = int(m)
            # Only consider plausible medium volumes (20–2500 mL typical)
            if 20 <= val <= 2500:
                volumes.append(val)
        except ValueError:
            continue

    # Remove duplicates while preserving order
    seen = set()
    unique_volumes = [v for v in volumes if not (v in seen or seen.add(v))]

    return unique_volumes






VOLUME_REGEX = r"([0-9]+(?:\.[0-9]+)?)\s*(?:ml|mL)?"
# VOLUME_REGEX = r"([0-9]+(?:\.[0-9]+)?)\s*ml\b"

VOLUME_MULTISTAGE_PATTERNS = [
    "stage",
    "phase",
    "tier",
    "acid",
    "buffer"
]


# Defining function to extract volumes
def extract_volumes(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    matches = re.findall(VOLUME_REGEX, text)
    volumes = [float(v) for v in matches]
    # remove unrealistic values
    volumes = [v for v in volumes if 50 <= v <= 3000]
    # volumes = [v for v in volumes if v <= 3000]
    return volumes


# Defining function to clean single-stage volume column
def classify_volume(text):
    if pd.isna(text):
        return None
    # do not assign volume if multistage
    if detect_multistage_volume(text):
        return None
    volumes = extract_volumes(text)
    if len(volumes) == 0:
        return None
    return volumes[0]


# Defining function to detect multi-stage volumes
def detect_multistage_volume(text):
    if pd.isna(text):
        return 0
    text = text.lower()
    return int(any(p in text for p in VOLUME_MULTISTAGE_PATTERNS))


# Defining function to extract multi-stage volumes
def extract_stage_volumes(text):
    if pd.isna(text):
        return pd.Series({
            "acid_volume": None,
            "buffer_volume": None
        })
    if not detect_multistage_volume(text):
        return pd.Series({
            "acid_volume": None,
            "buffer_volume": None
        })
    volumes = extract_volumes(text)
    acid = volumes[0] if len(volumes) >= 1 else None
    buffer = volumes[1] if len(volumes) >= 2 else None
    return pd.Series({
        "acid_volume": acid,
        "buffer_volume": buffer
    })









# 6. Function for Medium Volume Column

def extract_sampling_times(text):
    """
    Extract and standardize sampling times from a messy text entry.
    Returns:
        A list of numeric sampling times (in minutes).
        Returns [np.nan] if no valid times are found.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return [np.nan]
    if not isinstance(text, str):
        print(text, type(text))
        return [np.nan]

    clean_text = text.strip().lower()

    # Ignore irrelevant text
    if re.fullmatch(r"(develop a (dissolution|release) method|refer to (usp|fda's dissolution guidance)|n/?a|not applicable|none)", clean_text):
        return [np.nan]

    # Extract all numeric values (including decimals) followed by optional time units
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:minute|hour|hr|h|min|m)?s?", clean_text)

    # Convert to minutes
    times = []
    for m in matches:
        try:
            val = float(m)
            # Handle hours (e.g., "2 hours" -> 120 minutes)
            if re.search(r"(\d+(?:\.\d+)?)\s*(hour|hr|h)s?", clean_text, re.IGNORECASE):
                val *= 60
            # Only consider plausible sampling times (1–1440 minutes)
            if 1 <= val <= 1440:
                times.append(val)
        except ValueError:
            continue

    # Remove duplicates while preserving order
    seen = set()
    unique_times = [t for t in times if not (t in seen or seen.add(t))]

    return unique_times if unique_times else [np.nan]






TIME_REGEX = r"(\d*\.?\d+)\s*(hours?|hrs?|minutes?|mins?)?"

TIME_MULTISTAGE_PATTERNS = [
    "stage",
    "phase",
    "acid",
    "buffer"
]

STAGE_SPLIT_REGEX = r"(acid[^:]*:|buffer[^:]*:|phase\s*\d+[^:]*:)"


# Defining function to clean general text
def preprocess_sampling_text(text):
    """Clean general sampling text."""
    if pd.isna(text):
        return None
    text = str(text).lower()
    # remove drug name prefixes like "morphine sulfate:"
    text = re.sub(r"[a-z\s]+:\s*(acid|phase)", r"\1", text)
    # remove "no sampling"
    text = re.sub(r"phase\s*\d+\s*:\s*no sampling\.?", "", text)
    # remove instructions like "until 80% released"
    text = re.sub(r"until.*", "", text)
    return text    


# Defining function to detect multi-stage sampling methods
def detect_multistage_sampling(text):
    if pd.isna(text):
        return 0
    text = str(text).lower()
    return int(any(p in text for p in TIME_MULTISTAGE_PATTERNS))
    # return int(any(x in text for x in ["acid", "buffer", "phase"]))


# Defining function to extract times (convert hours → minutes)
def extract_times(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    matches = re.findall(TIME_REGEX, text)
    times = []
    for value, unit in matches:
        value = float(value)
        if unit and "hour" in unit:
            value *= 60
        times.append(value)
    return sorted(set(times))


# Defining function to clean single-stage sampling column
def classify_sampling_times(text):
    """Return times for single-stage sampling, None if multi-stage or empty."""
    if pd.isna(text):
        return None
    text = preprocess_sampling_text(text)
    if detect_multistage_sampling(text):
        return None
    times = extract_times(text)
    # if len(times) == 0:
    #     return None
    # return times
    return tuple(times) if times else None  # Convert to tuple
# def classify_sampling_times(text):
#     if pd.isna(text):
#         return None
#     text = preprocess_sampling_text(text)
#     if detect_multistage_sampling(text):
#         return None
#     times = extract_times(text)
#     return times if times else None


# Defining function to extract multi-stage sampling times
def extract_stage_sampling_times(text):
    """Return acid and buffer times as tuples."""
    if pd.isna(text) or not detect_multistage_sampling(text):
        return pd.Series({"acid_times": None, "buffer_times": None})
    # if pd.isna(text):
    #     return pd.Series({
    #         "acid_times": None,
    #         "buffer_times": None
    #     })
    # if not detect_multistage_sampling(text):
    #     return pd.Series({
    #         "acid_times": None,
    #         "buffer_times": None
    #     })
    text = str(text).lower()
    acid_part = text.split("acid")[1] if "acid" in text else None
    buffer_part = text.split("buffer")[1] if "buffer" in text else None
    acid_times = tuple(extract_times(acid_part)) if acid_part else None
    buffer_times = tuple(extract_times(buffer_part)) if buffer_part else None
    # acid_times = extract_times(acid_part) if acid_part else None
    # buffer_times = extract_times(buffer_part) if buffer_part else None
    return pd.Series({"acid_times": acid_times, "buffer_times": buffer_times})


# Defining function to apply to DataFrame to fully clean sampling times
def clean_sampling_times(df, col="Recommended Sampling Times (minutes)"):
    df = df.copy()
    # Detect multi-stage
    df["multi_stage_times"] = df[col].apply(detect_multistage_sampling)
    # Extract single-stage times
    df["SamplingTimes_Cleaned"] = df[col].apply(classify_sampling_times)
    # Extract acid/buffer times as tuples
    df[["acid_times", "buffer_times"]] = df[col].apply(extract_stage_sampling_times)
    return df


# --- Example Usage ---
# newer_clean_df = clean_sampling_times(newer_clean_df)

# Now you can safely do:
# Count of single column
# newer_clean_df["SamplingTimes_Cleaned"].value_counts(dropna=False)
# Count of multi-index
# newer_clean_df[["multi_stage_times", "acid_times", "buffer_times"]].value_counts(dropna=False)


# Defining function to compute the number of sampling times
def compute_num_sampling_points(row):
    # Use 'SamplingTimes_Cleaned' if not empty
    if isinstance(row["SamplingTimes_Cleaned"], (tuple, list)) and row["SamplingTimes_Cleaned"]:
        return len(row["SamplingTimes_Cleaned"])
    # Fallback to 'acid_times' and 'buffer_times'
    total_points = 0
    for col in ["acid_times", "buffer_times"]:
        if isinstance(row[col], (tuple, list)) and row[col]:
            total_points += len(row[col])
    return total_points if total_points > 0 else None


# Defining function to compute the maximum time across sampling times
def compute_max_sampling_time(row):
    times = []
    # Add 'SamplingTimes_Cleaned' if available
    if isinstance(row["SamplingTimes_Cleaned"], (tuple, list)) and row["SamplingTimes_Cleaned"]:
        times.extend(row["SamplingTimes_Cleaned"])
    # Add 'acid_times' and 'buffer_times'
    for col in ["acid_times", "buffer_times"]:
        if isinstance(row[col], (tuple, list)) and row[col]:
            times.extend(row[col])
    return max(times) if times else None





















def preprocess_fda_dissolution(df):
    """
    Complete preprocessing pipeline for FDA dissolution dataset.
    Returns a cleaned dataframe with engineered features.
    """
    df = df.copy()
    # -------------------------
    # MEDIUM FEATURES
    # -------------------------
    df["Medium_Type"] = df["Medium"].apply(classify_medium)
    df["pH"] = df["Medium"].apply(extract_ph)
    df["surfactant_presence"] = df["Medium"].apply(detect_surfactant)
    df[["surfactant_presence_refined", "surfactant_type"]] = (df["Medium"].apply(detect_surfactant_refined))
    df["multi_stage_medium"] = df["Medium"].apply(detect_multistage)
    df["multi_stage_medium_refined"] = df["Medium"].apply(detect_multistage_refined)
    # -------------------------
    # VOLUME FEATURES
    # -------------------------
    df["Volume_mL_clean"] = df["Volume (mL)"].apply(classify_volume)
    df["multi_stage_volume"] = df["Volume (mL)"].apply(detect_multistage_volume)
    df[["acid_volume_mL", "buffer_volume_mL"]] = (df["Volume (mL)"].apply(extract_stage_volumes))
    return df
    # clean_df = preprocess_fda_dissolution(newer_clean_df)


# def add_medium_features(df):
#     df["Medium_Type"] = df["Medium"].apply(classify_medium)
#     df["pH"] = df["Medium"].apply(extract_ph)
#     return df


# def add_volume_features(df):
#     df["Volume_mL_clean"] = df["Volume (mL)"].apply(classify_volume)
#     return df


# def preprocess_fda_dissolution(df):
#     df = df.copy()
#     df = add_medium_features(df)
#     df = add_volume_features(df)
#     return df
#     # clean_df = preprocess_fda_dissolution(newer_clean_df)








































# 7. Function for simulate dissolution curve and returning profile as a list in a new column

def simulate_dissolution_curve(row, time_points=None):
    """
    Simulate a dissolution profile for a single FDA test condition row.

    Parameters
    ----------
    row : pd.Series
        One row from your cleaned FDA dataset. Expected fields:
        - DosageForm_Cleaned
        - ReleaseType_Cleaned
        - Apparatus_Cleaned
        - Speed_RPM
        - MediumVolume_mL
        - MediumType_Cleaned
        - Temperature_C

    time_points : list or np.array, optional
        Time points (minutes) to compute % dissolved. Defaults to 0-120 min every 5 min.

    Returns
    -------
    pd.DataFrame
        Columns: ['Time_min', 'Dissolved_pct']
    """
    # Use provided time_points if available, otherwise default
    if time_points is None:
        time_points = np.arange(0, 125, 5)  # default: 0, 5, 10, ..., 120
    else:
        time_points = np.array(time_points)  # ensure numpy array

    # Extract inputs with fallback defaults
    release_type = str(row.get("ReleaseType_Cleaned", "Immediate Release")).lower()
    speed = float(row.get("SpeedRPM_Cleaned", 75) or 75)         # default RPM
    volume = float(row.get("MediumVolume_Cleaned", 900) or 900)  # default mL
    temp = float(row.get("Temperature_C", 37) or 37)             # default °C
    
    # --- Step 1: Choose kinetic constant based on release type ---
    # These constants are arbitrary examples; you can calibrate later
    if "immediate" in release_type:
        k = 0.8   # fast dissolution
    elif "extended" in release_type:
        k = 0.2   # slow dissolution
    elif "delayed" in release_type:
        k = 0.5   # moderate dissolution
    else:
        k = 0.6   # default

    # Optionally adjust k based on RPM (more agitation → faster dissolution)
    k *= (1 + (speed - 50)/200)  # e.g., 50 RPM baseline

    # Optionally adjust k based on volume (smaller volume → slower dissolution)
    k *= (volume / 900) ** 0.1  # mild effect


    # --- Step 2: Compute % dissolved at each time point ---
    # Using simple first-order kinetics: C(t) = 100 * (1 - exp(-k * t))
    dissolved = 100 * (1 - np.exp(-k * time_points / 60))  # divide by 60 to scale k to minutes

    # Clip to 0–100%
    dissolved = np.clip(dissolved, 0, 100)

    # --- Step 3: Add small noise to simulate experimental variation ---
    noise = np.random.normal(0, 2, size=len(dissolved))  # ±2% noise
    dissolved = np.clip(dissolved + noise, 0, 100)

    # --- Step 4: Return as DataFrame ---
    return pd.DataFrame({
        "Time_min": time_points,
        "Dissolved_pct": dissolved
    })


# Fallback for if some rows have missing or invalid time points
def get_time_points(row):
    time_points = row.get("SamplingTimes_Cleaned")
    if time_points is None or not isinstance(time_points, (list, np.ndarray)):
        return np.arange(0, 125, 5)
    return np.array(time_points)











    

    






def extract_agitation_speed(text, apparatus):
    """
    Simplify the Speed (RPM) column into clean numerical and standardized values.
    Extract agitation speed (numeric) from text and validate against apparatus type.
    Returns np.nan if invalid or nonmatching.
    """

    # 1. Handle missing or non-string inputs
    if not isinstance(text, str) or not isinstance(apparatus, str):
        return np.nan

    text_lower = text.lower().strip()
    apparatus = apparatus.strip()

    # 2. Ignore irrelevant cases
    if any(word in text_lower for word in ["flow", "ml/min", "stroke", "orbit", "chew"]):
        return np.nan

    # 3. Extract numeric value and unit (if any)
    match = re.search(r'(\d+(?:\.\d+)?)\s*(rpm|dpm|cpm|cycles|dips)', text_lower)
    if not match:
        return np.nan  # no recognizable agitation term

    speed_value = float(match.group(1))
    unit = match.group(2)

    # 4. Check valid unit–apparatus match
    valid_units = APPARATUS_UNITS.get(apparatus, [])
    if unit not in valid_units:
        return np.nan

    # 5. Return cleaned speed
    return speed_value



def extract_rpm(text):
    """
    Simplify the Speed (RPM) column into clean numerical and standardized values.
    """
    if not isinstance(text, str) or text.strip() == "":
        return np.nan
    text_lower = text.lower()

    # Ignore irrelevant cases
    if any(x in text_lower for x in ["flow", "ml/min", "chew", "stroke", "orbit"]):
        return np.nan

    # Extract first numeric value that precedes rpm/dpm/cycles/dips
    match = re.search(r'(\d+)\s*(?:rpm|dpm|cycles|dips)', text_lower)
    if match:
        return float(match.group(1))

    # Fallback: just a number (e.g. "50")
    match = re.match(r'^\d+$', text_lower.strip())
    if match:
        return float(match.group(0))

    # Multiple speeds in one cell — split and take first numeric
    nums = re.findall(r'\d+', text_lower)
    if len(nums) == 1:
        return float(nums[0])
    elif len(nums) > 1:
        # optional: duplicate row logic; for now, take the first
        return float(nums[0])
    else:
        return np.nan




