import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
DESCRIPTION:

Validate synthetic education-location assignments.

The validation checks:

1. Population / assignment conservation
2. Expected vs assigned students per school
3. Expected vs assigned students per university
4. Home-to-education distance distributions
5. Assigned distance vs nearest eligible school distance
6. Origin municipality -> destination municipality flows
7. Spatial distribution of education locations
8. Summary statistics

This stage is intended to run on the FULL synthetic population.
Therefore absolute counts are compared directly against expected
student counts where appropriate.

The stage does not modify the synthetic population.
"""


def configure(context):

    context.stage("synthesis.output")
    context.stage("seville.gravity.od_zones")
    context.stage("seville.data.education.merged")
    context.stage("seville.data.education.universities")

    context.config("analysis_path")
    context.config("output_path")
    context.config("output_prefix")

    context.config("education_graduation_age")


# ==============================================================================
# HELPERS
# ==============================================================================

def education_type(age, graduation_ages):

    if age <= graduation_ages["kindergarten"]:
        return "kindergarten"

    elif age <= graduation_ages["elementary"]:
        return "elementary"

    elif age <= graduation_ages["highschool"]:
        return "highschool"

    return "university"


def fix_commune(df):

    df = df.copy()

    df["commune_id"] = df["commune_id"].astype(str)

    # Municipality code: PPMMM
    df["commune_id_new"] = df["commune_id"].str[:5]

    # Seville: use district PPMMMDD
    SEVILLE_MUN_CODE = "41091"

    mask = df["commune_id"].str[:5] == SEVILLE_MUN_CODE

    df.loc[mask, "commune_id_new"] = (
        df.loc[mask, "commune_id"].str[:7]
    )

    df["commune_id"] = df["commune_id_new"]

    return df.drop(columns=["commune_id_new"])


def safe_relative_error(assigned, expected):

    if expected == 0:
        return np.nan

    return (assigned - expected) / expected


def print_section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ==============================================================================
# MAIN
# ==============================================================================

def execute(context):

    analysis_path = context.config("analysis_path")
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")

    graduation_ages = context.config(
        "education_graduation_age"
    )

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================

    print_section("LOADING DATA")

    df_persons = pd.read_csv(
        f"{output_path}/{output_prefix}persons.csv",
        encoding="latin1",
        sep=";"
    )

    gdf_homes = gpd.read_file(
        f"{output_path}/{output_prefix}homes.gpkg"
    )

    gdf_activities = gpd.read_file(
        f"{output_path}/{output_prefix}activities.gpkg"
    )

    df_zones, _, _ = context.stage(
        "seville.gravity.od_zones"
    )

    df_zones = df_zones[
        ["macrozone_id", "geometry"]
    ].copy()

    df_expected_schools = context.stage(
        "seville.data.education.merged"
    ).copy()

    df_expected_universities = context.stage(
        "seville.data.education.universities"
    ).copy()

    print(f"Persons:    {len(df_persons):,}")
    print(f"Homes:      {len(gdf_homes):,}")
    print(f"Activities: {len(gdf_activities):,}")


    # ==========================================================================
    # BASIC PERSON PREPARATION
    # ==========================================================================

    df_persons["education_type"] = (
        df_persons["age"].apply(
            lambda age: education_type(
                age,
                graduation_ages
            )
        )
    )

    # Persons with education activities
    education_activities = gdf_activities[
        gdf_activities["purpose"] == "education"
    ].copy()

    education_activities = education_activities.merge(
        df_persons[
            [
                "person_id",
                "age",
                "household_id",
                "education_type"
            ]
        ],
        on="person_id",
        how="left",
        suffixes=("", "_person")
    )


    # ==========================================================================
    # 1. ASSIGNMENT CONSERVATION
    # ==========================================================================

    print_section("1. ASSIGNMENT CONSERVATION")

    eligible = df_persons[
        df_persons["education_type"].notna()
    ].copy()

    eligible_counts = (
        eligible
        .groupby("education_type")
        .size()
        .rename("eligible_students")
    )

    assigned_counts = (
        education_activities
        .groupby("education_type")["person_id"]
        .nunique()
        .rename("assigned_students")
    )

    assignment_frequency = (
        education_activities
        .groupby("person_id")
        .size()
    )

    multiple_assigned = (
        assignment_frequency > 1
    ).sum()

    conservation = (
        pd.concat(
            [
                eligible_counts,
                assigned_counts
            ],
            axis=1
        )
        .fillna(0)
        .reset_index()
    )

    conservation["unassigned_students"] = (
        conservation["eligible_students"]
        - conservation["assigned_students"]
    )

    conservation["assignment_rate"] = (
        conservation["assigned_students"]
        / conservation["eligible_students"]
    )

    conservation["assignment_rate"] = (
        conservation["assignment_rate"]
        .replace([np.inf, -np.inf], np.nan)
    )

    conservation.to_csv(
        f"{analysis_path}/education_assignment_conservation.csv",
        index=False
    )

    print(conservation.to_string(index=False))

    print(
        f"\nStudents with multiple education activities: "
        f"{multiple_assigned:,}"
    )


    # ==========================================================================
    # 2. SCHOOL ASSIGNMENT DATA
    # ==========================================================================

    print_section("2. SCHOOL CAPACITY VALIDATION")

    school_students = education_activities[
        education_activities["education_type"] != "university"
    ].copy()

    # Keep only expected school types
    school_students = school_students[
        school_students["education_type"].isin(
            [
                "kindergarten",
                "elementary",
                "highschool"
            ]
        )
    ].copy()

    # Geometry key
    school_students["location_key"] = (
        school_students.geometry.apply(
            lambda geometry: geometry.wkb
        )
    )

    # Identify synthetic schools
    gdf_schools = (
        school_students[
            [
                "location_key",
                "geometry",
                "education_type"
            ]
        ]
        .drop_duplicates(
            [
                "location_key",
                "education_type"
            ]
        )
        .reset_index(drop=True)
    )

    gdf_schools["school_id"] = np.arange(
        len(gdf_schools)
    )

    gdf_schools = gpd.GeoDataFrame(
        gdf_schools,
        geometry="geometry",
        crs=school_students.crs
    )

    school_students = school_students.merge(
        gdf_schools[
            [
                "location_key",
                "education_type",
                "school_id"
            ]
        ],
        on=[
            "location_key",
            "education_type"
        ],
        how="left"
    )

    # --------------------------------------------------------------------------
    # Expected schools
    # --------------------------------------------------------------------------

    df_expected_schools = df_expected_schools[
        ~df_expected_schools["education_type"].isin(
            ["university", "other"]
        )
    ].copy()

    df_expected_schools = (
        df_expected_schools
        .to_crs(school_students.crs)
    )

    df_expected_schools["location_key"] = (
        df_expected_schools.geometry.apply(
            lambda geometry: geometry.wkb
        )
    )

    expected_school_counts = (
        df_expected_schools
        .groupby(
            [
                "location_key",
                "education_type"
            ],
            as_index=False
        )["students"]
        .sum()
        .rename(
            columns={
                "students": "expected_students"
            }
        )
    )

    # --------------------------------------------------------------------------
    # Assigned schools
    # --------------------------------------------------------------------------

    assigned_school_counts = (
        school_students
        .groupby(
            [
                "location_key",
                "education_type",
                "school_id"
            ],
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "assigned_students"
            }
        )
    )

    # --------------------------------------------------------------------------
    # Combine
    # --------------------------------------------------------------------------

    school_stats = expected_school_counts.merge(
        assigned_school_counts,
        on=[
            "location_key",
            "education_type"
        ],
        how="outer"
    )

    school_stats["expected_students"] = (
        school_stats["expected_students"]
        .fillna(0)
    )

    school_stats["assigned_students"] = (
        school_stats["assigned_students"]
        .fillna(0)
    )

    school_stats["school_id"] = (
        school_stats["school_id"]
        .fillna(-1)
        .astype(int)
    )

    school_stats["difference"] = (
        school_stats["assigned_students"]
        - school_stats["expected_students"]
    )

    school_stats["absolute_error"] = (
        school_stats["difference"].abs()
    )

    school_stats["relative_error"] = (
        school_stats.apply(
            lambda row: safe_relative_error(
                row["assigned_students"],
                row["expected_students"]
            ),
            axis=1
        )
    )

    school_stats["assigned_to_expected_ratio"] = (
        school_stats["assigned_students"]
        / school_stats["expected_students"]
    )

    school_stats["assigned_to_expected_ratio"] = (
        school_stats["assigned_to_expected_ratio"]
        .replace([np.inf, -np.inf], np.nan)
    )

    school_stats.to_csv(
        f"{analysis_path}/education_school_capacity.csv",
        index=False
    )

    # Summary by education type

    school_capacity_summary = (
        school_stats
        .groupby("education_type")
        .agg(
            expected_students=("expected_students", "sum"),
            assigned_students=("assigned_students", "sum"),
            mean_absolute_error=("absolute_error", "mean"),
            total_absolute_error=("absolute_error", "sum"),
            median_assignment_ratio=(
                "assigned_to_expected_ratio",
                "median"
            )
        )
        .reset_index()
    )

    school_capacity_summary["difference"] = (
        school_capacity_summary["assigned_students"]
        - school_capacity_summary["expected_students"]
    )

    school_capacity_summary["WAPE"] = (
        school_capacity_summary["total_absolute_error"]
        / school_capacity_summary["expected_students"]
    )

    school_capacity_summary.to_csv(
        f"{analysis_path}/education_school_capacity_summary.csv",
        index=False
    )

    print(school_capacity_summary.to_string(index=False))


    # ==========================================================================
    # 3. UNIVERSITY CAPACITY VALIDATION
    # ==========================================================================

    print_section("3. UNIVERSITY CAPACITY VALIDATION")

    university_students = education_activities[
        education_activities["education_type"] == "university"
    ].copy()

    university_students["location_key"] = (
        university_students.geometry.apply(
            lambda geometry: geometry.wkb
        )
    )

    # Assigned university counts
    assigned_universities = (
        university_students
        .groupby("location_key")
        .size()
        .reset_index(
            name="assigned_students"
        )
    )

    # Expected university data
    df_expected_universities = (
        df_expected_universities
        .to_crs(university_students.crs)
        .copy()
    )

    df_expected_universities["location_key"] = (
        df_expected_universities.geometry.apply(
            lambda geometry: geometry.wkb
        )
    )

    expected_universities = (
        df_expected_universities
        .groupby(
            [
                "location_key",
                "location_name"
            ],
            as_index=False
        )["students"]
        .sum()
        .rename(
            columns={
                "students": "expected_students"
            }
        )
    )

    university_stats = expected_universities.merge(
        assigned_universities,
        on="location_key",
        how="outer"
    )

    university_stats["expected_students"] = (
        university_stats["expected_students"]
        .fillna(0)
    )

    university_stats["assigned_students"] = (
        university_stats["assigned_students"]
        .fillna(0)
    )

    university_stats["location_name"] = (
        university_stats["location_name"]
        .fillna("UNMATCHED")
    )

    university_stats["difference"] = (
        university_stats["assigned_students"]
        - university_stats["expected_students"]
    )

    university_stats["absolute_error"] = (
        university_stats["difference"].abs()
    )

    university_stats["relative_error"] = (
        university_stats.apply(
            lambda row: safe_relative_error(
                row["assigned_students"],
                row["expected_students"]
            ),
            axis=1
        )
    )

    university_stats["assigned_to_expected_ratio"] = (
        university_stats["assigned_students"]
        / university_stats["expected_students"]
    )

    university_stats["assigned_to_expected_ratio"] = (
        university_stats["assigned_to_expected_ratio"]
        .replace([np.inf, -np.inf], np.nan)
    )

    university_stats.to_csv(
        f"{analysis_path}/education_university_capacity.csv",
        index=False
    )

    print(
        university_stats[
            [
                "location_name",
                "expected_students",
                "assigned_students",
                "difference",
                "relative_error"
            ]
        ].to_string(index=False)
    )


    # ==========================================================================
    # 4. HOME-TO-EDUCATION DISTANCES
    # ==========================================================================

    print_section("4. HOME-TO-EDUCATION DISTANCES")

    education_students = education_activities.merge(
        gdf_homes[
            [
                "household_id",
                "geometry"
            ]
        ].rename(
            columns={
                "geometry": "home_geometry"
            }
        ),
        on="household_id",
        how="left"
    )

    education_students = gpd.GeoDataFrame(
        education_students,
        geometry="geometry",
        crs=education_activities.crs
    )

    # Reproject to metric CRS
    distance_crs = gdf_homes.estimate_utm_crs()

    education_students = education_students.to_crs(
        distance_crs
    )

    education_students["home_geometry"] = (
        gpd.GeoSeries(
            education_students["home_geometry"],
            crs=gdf_homes.crs
        )
        .to_crs(distance_crs)
    )

    valid_distance = (
        education_students["home_geometry"].notna()
        & education_students.geometry.notna()
    )

    education_students["distance_m"] = np.nan

    education_students.loc[
        valid_distance,
        "distance_m"
    ] = (
        education_students.loc[
            valid_distance,
            "home_geometry"
        ]
        .distance(
            education_students.loc[
                valid_distance,
                "geometry"
            ]
        )
    )

    education_students["distance_km"] = (
        education_students["distance_m"] / 1000
    )

    distance_summary = (
        education_students
        .dropna(subset=["distance_km"])
        .groupby("education_type")["distance_km"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.90),
            p95=lambda x: x.quantile(0.95),
            maximum="max"
        )
        .reset_index()
    )

    distance_summary.to_csv(
        f"{analysis_path}/education_distance_summary.csv",
        index=False
    )

    print(distance_summary.to_string(index=False))


    # ==========================================================================
    # GRAPH 1 — DISTANCE DISTRIBUTIONS
    # ==========================================================================

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for education_type_name, subset in (
        education_students
        .dropna(subset=["distance_km"])
        .groupby("education_type")
    ):

        ax.hist(
            subset["distance_km"],
            bins=50,
            alpha=0.5,
            density=True,
            label=education_type_name
        )

    ax.set_xlabel(
        "Home-to-education distance (km)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Synthetic Home-to-Education Distance Distribution"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        f"{analysis_path}/education_distance_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ==========================================================================
    # 5. NEAREST SCHOOL ANALYSIS
    # ==========================================================================

    print_section("5. NEAREST SCHOOL ANALYSIS")

    # Only schools, not universities
    nearest_students = education_students[
        education_students["education_type"].isin(
            [
                "kindergarten",
                "elementary",
                "highschool"
            ]
        )
    ].copy()

    nearest_schools = gdf_schools.to_crs(
        distance_crs
    )

    # Use nearest school of the same education type.
    #
    # This is deliberately done per education type because comparing
    # a primary student against a high school is not meaningful.

    nearest_values = []

    for education_type_name in (
        nearest_students["education_type"]
        .dropna()
        .unique()
    ):

        students_subset = nearest_students[
            nearest_students["education_type"]
            == education_type_name
        ].copy()

        schools_subset = nearest_schools[
            nearest_schools["education_type"]
            == education_type_name
        ].copy()

        if len(students_subset) == 0:
            continue

        if len(schools_subset) == 0:
            continue

        # Spatial nearest neighbour
        nearest = gpd.sjoin_nearest(
            students_subset[
                [
                    "person_id",
                    "home_geometry",
                    "geometry"
                ]
            ].set_geometry("home_geometry"),
            schools_subset[
                [
                    "school_id",
                    "geometry"
                ]
            ],
            how="left",
            distance_col="nearest_distance_m"
        )

        nearest_values.append(
            nearest[
                [
                    "person_id",
                    "nearest_distance_m"
                ]
            ].assign(
                education_type=education_type_name
            )
        )

    if nearest_values:

        nearest_distances = pd.concat(
            nearest_values,
            ignore_index=True
        )

        nearest_distances["nearest_distance_km"] = (
            nearest_distances["nearest_distance_m"]
            / 1000
        )

        education_students = education_students.merge(
            nearest_distances[
                [
                    "person_id",
                    "education_type",
                    "nearest_distance_km"
                ]
            ],
            on=[
                "person_id",
                "education_type"
            ],
            how="left"
        )

        education_students["distance_ratio"] = (
            education_students["distance_km"]
            / education_students["nearest_distance_km"]
        )

        nearest_summary = (
            education_students
            .dropna(subset=["distance_ratio"])
            .groupby("education_type")[
                "distance_ratio"
            ]
            .agg(
                median="median",
                p75=lambda x: x.quantile(0.75),
                p90=lambda x: x.quantile(0.90),
                mean="mean"
            )
            .reset_index()
        )

        nearest_summary.to_csv(
            f"{analysis_path}/education_nearest_school_summary.csv",
            index=False
        )

        print(
            nearest_summary.to_string(index=False)
        )


    # ==========================================================================
    # 6. HOME MUNICIPALITY -> EDUCATION MUNICIPALITY
    # ==========================================================================

    print_section("6. ORIGIN -> DESTINATION MUNICIPALITY")

    # Home municipality comes from the home zone attached to the household.
    #
    # We assume gdf_homes contains commune_id. If it does not, this section
    # should be connected to the appropriate home-zone stage instead.

    if "commune_id" in gdf_homes.columns:

        home_communes = gdf_homes[
            [
                "household_id",
                "commune_id"
            ]
        ].copy()

        home_communes = fix_commune(
            home_communes
        )

        education_students = education_students.merge(
            home_communes,
            on="household_id",
            how="left",
            suffixes=("", "_home")
        )

        # Education location municipality
        #
        # Prefer commune_id on the activity data if available.
        if "commune_id" in education_activities.columns:

            education_students["destination_commune"] = (
                education_students["commune_id"]
            )

        elif "commune_id" in gdf_schools.columns:

            education_students["destination_commune"] = (
                education_students["commune_id"]
            )

        else:

            education_students["destination_commune"] = np.nan

        education_students = fix_commune(
            education_students
        )

        if "destination_commune" in education_students.columns:

            flow = (
                education_students
                .dropna(
                    subset=[
                        "commune_id",
                        "destination_commune"
                    ]
                )
                .groupby(
                    [
                        "commune_id",
                        "destination_commune",
                        "education_type"
                    ]
                )
                .size()
                .reset_index(
                    name="assigned_students"
                )
            )

            flow.to_csv(
                f"{analysis_path}/education_origin_destination.csv",
                index=False
            )

            print(
                f"OD pairs: {len(flow):,}"
            )


    # ==========================================================================
    # 7. SCHOOL CAPACITY GRAPH
    # ==========================================================================

    print_section("7. GENERATING PLOTS")

    school_plot = school_stats[
        school_stats["expected_students"] > 0
    ].copy()

    if len(school_plot) > 0:

        fig, ax = plt.subplots(
            figsize=(9, 9)
        )

        for education_type_name, subset in (
            school_plot.groupby("education_type")
        ):

            ax.scatter(
                subset["expected_students"],
                subset["assigned_students"],
                alpha=0.7,
                label=education_type_name
            )

        max_value = max(
            school_plot["expected_students"].max(),
            school_plot["assigned_students"].max()
        )

        ax.plot(
            [0, max_value],
            [0, max_value],
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.6
        )

        ax.set_xlabel(
            "Expected students"
        )

        ax.set_ylabel(
            "Assigned students"
        )

        ax.set_title(
            "School Expected vs Assigned Students"
        )

        ax.legend()
        ax.grid(alpha=0.25)

        plt.tight_layout()

        plt.savefig(
            f"{analysis_path}/school_expected_vs_assigned.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


    # ==========================================================================
    # 8. ASSIGNMENT RATIO DISTRIBUTION
    # ==========================================================================

    ratio_data = school_stats[
        school_stats["assigned_to_expected_ratio"]
        .notna()
    ].copy()

    if len(ratio_data) > 0:

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        for education_type_name, subset in (
            ratio_data.groupby("education_type")
        ):

            ax.hist(
                subset["assigned_to_expected_ratio"],
                bins=40,
                alpha=0.5,
                label=education_type_name
            )

        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=1,
            color="black"
        )

        ax.set_xlabel(
            "Assigned / expected students"
        )

        ax.set_ylabel(
            "Number of schools"
        )

        ax.set_title(
            "School Assignment Ratio Distribution"
        )

        ax.legend()
        ax.grid(alpha=0.25)

        plt.tight_layout()

        plt.savefig(
            f"{analysis_path}/school_assignment_ratio_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


    # ==========================================================================
    # 9. SUMMARY
    # ==========================================================================

    print_section("VALIDATION SUMMARY")

    total_eligible = len(eligible)

    total_assigned = (
        education_activities["person_id"]
        .nunique()
    )

    total_unassigned = (
        total_eligible
        - total_assigned
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "eligible education students",
                "value": total_eligible
            },
            {
                "metric": "assigned education students",
                "value": total_assigned
            },
            {
                "metric": "unassigned education students",
                "value": total_unassigned
            },
            {
                "metric": "assignment rate",
                "value": (
                    total_assigned / total_eligible
                    if total_eligible > 0
                    else np.nan
                )
            },
            {
                "metric": "students with multiple education assignments",
                "value": multiple_assigned
            }
        ]
    )

    summary.to_csv(
        f"{analysis_path}/education_validation_summary.csv",
        index=False
    )

    print(
        summary.to_string(index=False)
    )

    print()
    print(
        f"Validation outputs written to: {analysis_path}"
    )

    return summary
