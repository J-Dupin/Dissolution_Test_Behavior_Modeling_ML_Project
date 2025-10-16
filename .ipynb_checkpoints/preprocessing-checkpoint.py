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

def clean_apparatus_column(df, col='USP Apparatus'):
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

    df['Apparatus_Cleaned'] = df[col].apply(extract_apparatus)
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

    # 1️⃣ Extract all numbers with valid units
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(rpm|dpm|cpm|cycles|dips)', text_lower)
    valid_units = APPARATUS_UNITS.get(apparatus, [])
    filtered_numbers = [float(val) for val, unit in matches if unit in valid_units]

    # 2️⃣ Fallback: if no matches, extract any plain numbers
    if not filtered_numbers:
        plain_numbers = re.findall(r'\d+', text_lower)
        # Only use as fallback if apparatus expects a numeric speed
        if valid_units and plain_numbers:
            filtered_numbers = [float(n) for n in plain_numbers]

    # 3️⃣ Duplicate row for each valid speed
    new_rows = []
    for speed in filtered_numbers:
        new_row = row.copy()
        new_row['Agitation_Speed'] = speed
        new_rows.append(new_row)

    return new_rows






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

# ----------------- Example usage -----------------
# Assuming df_cleaned is your processed FDA DataFrame
# row = df_cleaned.iloc[0]
# df_curve = simulate_dissolution_curve(row)
# print(df_curve.head())











    

    






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




