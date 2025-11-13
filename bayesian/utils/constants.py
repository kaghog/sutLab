INDIVIDUAL_COLS = [
    "Sex", "Age", "Marital_status", "Education",
    "Has_vehicle", "Employment_status"
]
HOUSEHOLD_COLS = [
    "Residence_type", "Num_rooms", "Floor_material",
    "Wall_material", "Roof_material", "Wealth_index", "Household_size"
]
SPOUSE = "Wife or husband"
CHILDREN = ["Son/daughter", "Adopted/foster child"]
OTHER_MEMBERS = [
    "Other relative", "Parent", "Parent-in-law", "Grandchild",
    "Not related", "Brother/sister", "Son/daughter-in-law"
]

# expected household types (labels as they appear in data) and their short codes
HOUSEHOLD_TYPES_MAP = {
    "Single": "SNG",
    "Couple with no children": "CWNC",
    "Couple with children": "CWC",
    "Couple with children and Other household members": "CWCAO",
    "Couple with no children and Other household members": "CWNCAO",
    "Single parent with children": "SPWC",
    "Single parent with children and Other household members": "SPWCAO",
    "Head with Other household members": "HWO",
    "Other": "OTH"
}

# required columns: keep encoded names and rename into concise internal names
REQUIRED_COLUMNS = [
    "Person ID", "Household ID", "Region", "Household type",
    "Sex of household member_encoded",
    "Age_bin of household members_encoded",
    "Current marital status_encoded",
    "Educational attainment_encoded",
    "Has vehicle_encoded",
    "Employment status_encoded",
    "Type of place of residence_encoded",
    "Relationship to head",
    "Number of rooms_bin_encoded",
    "Main floor material_encoded",
    "Main wall material_encoded",
    "Main roof material_encoded",
    "Wealth index combined_encoded",
    "Household Size_bin_encoded",
]

RENAME_COLUMNS = {
    "Sex of household member_encoded": "Sex",
    "Age_bin of household members_encoded": "Age",
    "Current marital status_encoded": "Marital_status",
    "Educational attainment_encoded": "Education",
    "Has vehicle_encoded": "Has_vehicle",
    "Employment status_encoded": "Employment_status",
    "Type of place of residence_encoded": "Residence_type",
    "Number of rooms_bin_encoded": "Num_rooms",
    "Main floor material_encoded": "Floor_material",
    "Main wall material_encoded": "Wall_material",
    "Main roof material_encoded": "Roof_material",
    "Wealth index combined_encoded": "Wealth_index",
    "Household Size_bin_encoded": "Household_size",
}

ENCODING_LEGEND = {
    "Sex": {
         "Male": 1,
         "Female": 2
    },
    "Age": {
         "<15": 1,
         "15-24": 2,
         "25-34": 3,
         "35-44": 4,
         "45-54": 5,
         "55-64": 6,
         "65+": 7
    },
    "Marital_status": {
         "Never married": 1,
         "Married": 2,
         "Not living together": 3,
         "Divorced": 4,
         "Widowed": 5
    },
    "Education": {
         "No education": 1,
         "Incomplete primary": 2,
         "Complete primary": 3,
         "Incomplete secondary": 4,
         "Complete secondary": 5,
         "Higher": 6
    },
    "Has_vehicle": {
         "No": 0,
         "Yes": 1
    },
    "Employment_status" : {
         "Not applicable": 0,
         "Employed": 1,
         "Unemployed": 2,
         "Not in labor force": 3
    },
    "Residence_type": {
         "Urban": 1,
         "Rural": 2
    },
    "Num_rooms": {
         "1-4": 1,
         "5": 2,
         "6": 3,
         "7": 4,
         "8 or more": 5
    },
    "Floor_material": {
         "Carpet": 1,
         "Ceramic tiles": 2,
         "Cement": 3,
         "Dung": 4,
         "Earth/sand": 5,
         "Parquet or polished wood": 6,
         "Vinyl or asphalt strips": 7,
         "Wood planks": 8,
         "Other": 9
    },
    "Wall_material": {
         "Bamboo with mud": 1,
         "Bricks": 2,
         "Cane/palm/trunks": 3,
         "Cardboard": 4,
         "Cement": 5,
         "Cement blocks": 6,
         "Covered adobe": 7,
         "Dirt": 8,
         "Iron sheets": 9,
         "No walls": 10,
         "Plywood": 11,
         "Reused wood": 12,
         "Stone with lime/cement": 13,
         "Stone with mud": 14,
         "Uncovered adobe": 15,
         "Wood planks/shingles": 16,
         "Other": 17
    },
    "Roof_material": {
         "Asbestos sheet": 1,
         "Calamine/cement fiber": 2,
         "Cardboard": 3,
         "Cement": 4,
         "Ceramic tiles": 5,
         "Iron sheets/Metal": 6,
         "Palm/bamboo": 7,
         "Roofing shingles": 8,
         "Rustic mat": 9,
         "Sod/mud/dung": 10,
         "Thatch/grass/makuti": 11,
         "Tin cans": 12,
         "Wood": 13,
         "Wood planks": 14,
         "Other": 15,
         "No roof": 16
    },
    "Wealth_index": {
         "Poorest": 1,
         "Poorer": 2,
         "Middle": 3,
         "Richer": 4,
         "Richest": 5
    },
     "Household_size": {
         "1": 1,
         "2": 2,
         "3": 3,
         "4": 4,
         "5 or more": 5
    },
       "Sex of household member": {
         "Male": 1,
         "Female": 2
    },
    "Age_bin of household members": {
         "<15": 1,
         "15-24": 2,
         "25-34": 3,
         "35-44": 4,
         "45-54": 5,
         "55-64": 6,
         "65+": 7
    },
    "Current marital status": {
         "Never married": 1,
         "Married": 2,
         "Not living together": 3,
         "Divorced": 4,
         "Widowed": 5
    },
    "Educational attainment": {
         "No education": 1,
         "Incomplete primary": 2,
         "Complete primary": 3,
         "Incomplete secondary": 4,
         "Complete secondary": 5,
         "Higher": 6
    },
    "Has bicycle": {
         "No": 0,
         "Yes": 1
    },
    "Has motorcycle/scooter": {
         "No": 0,
         "Yes": 1
    },
    "Has car/truck": {
         "No": 0,
         "Yes": 1
    },
    "Has vehicle": {
         "No": 0,
         "Yes": 1
    },
    "Employment status" : {
         "Not applicable": 0,
         "Employed": 1,
         "Unemployed": 2,
         "Not in labor force": 3
    },
    "Type of place of residence": {
         "Urban": 1,
         "Rural": 2
    },
    "Sex of head of household": {
         "Male": 1,
         "Female": 2
    },
    "Age_bin of head of household": {
         "<15": 1,
         "15-24": 2,
         "25-34": 3,
         "35-44": 4,
         "45-54": 5,
         "55-64": 6,
         "65+": 7
    },
    "Household Size_bin": {
         "1": 1,
         "2": 2,
         "3": 3,
         "4": 4,
         "5 or more": 5
    },
    "Number of rooms_bin": {
         "1-4": 1,
         "5": 2,
         "6": 3,
         "7": 4,
         "8 or more": 5
    },
    "Main floor material": {
         "Carpet": 1,
         "Ceramic tiles": 2,
         "Cement": 3,
         "Dung": 4,
         "Earth/sand": 5,
         "Parquet or polished wood": 6,
         "Vinyl or asphalt strips": 7,
         "Wood planks": 8,
         "Other": 9
    },
    "Main wall material": {
         "Bamboo with mud": 1,
         "Bricks": 2,
         "Cane/palm/trunks": 3,
         "Cardboard": 4,
         "Cement": 5,
         "Cement blocks": 6,
         "Covered adobe": 7,
         "Dirt": 8,
         "Iron sheets": 9,
         "No walls": 10,
         "Plywood": 11,
         "Reused wood": 12,
         "Stone with lime/cement": 13,
         "Stone with mud": 14,
         "Uncovered adobe": 15,
         "Wood planks/shingles": 16,
         "Other": 17
    },
    "Main roof material": {
         "Asbestos sheet": 1,
         "Calamine/cement fiber": 2,
         "Cardboard": 3,
         "Cement": 4,
         "Ceramic tiles": 5,
         "Iron sheets/Metal": 6,
         "Palm/bamboo": 7,
         "Roofing shingles": 8,
         "Rustic mat": 9,
         "Sod/mud/dung": 10,
         "Thatch/grass/makuti": 11,
         "Tin cans": 12,
         "Wood": 13,
         "Wood planks": 14,
         "Other": 15,
         "No roof": 16
    },
    "Wealth index combined": {
         "Poorest": 1,
         "Poorer": 2,
         "Middle": 3,
         "Richer": 4,
         "Richest": 5
    }
}