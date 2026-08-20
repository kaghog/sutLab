import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from shapely.geometry import Point



def configure(context):
    context.config("output_path")
    context.config("data_path")
    context.config("analysis_path")
    context.config("output_prefix")
    context.config("debug")

    context.stage("synthesis.output")
    context.stage("seville.gravity.od_zones")
    context.config("education_graduation_age")
    context.stage("synthesis.locations.education")

def execute(context):


    if context.config("debug") == False:
        from debug import snapshot
        snapshot(context,
            configs=["output_path", "data_path", "analysis_path", "output_prefix", "education_graduation_age"],
            stages=["seville.gravity.od_zones", "seville.data.education.merged", "seville.data.education.universities"]
            )


    # ==============================================================================
    # LOAD DATA
    # ==============================================================================

    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")
    analysis_path = context.config("analysis_path")

    df_persons = pd.read_csv("%s/%spersons.csv" % (output_path, output_prefix),encoding="latin1",sep=";")

    gdf_homes = gpd.read_file("%s/%shomes.gpkg" % (output_path, output_prefix))

    gdf_activities = gpd.read_file("%s/%sactivities.gpkg" % (output_path, output_prefix))

    df_zones, _, _ = context.stage("seville.gravity.od_zones")

    df_zones = df_zones[["macrozone_id","geometry"]].copy()


    # ==============================================================================
    # EDUCATION AGE THRESHOLDS
    # ==============================================================================

    graduation_ages = context.config("education_graduation_age")

    kindergarten_age = graduation_ages["kindergarten"]
    elementary_age = graduation_ages["elementary"]
    highschool_age = graduation_ages["highschool"]


    # ==============================================================================
    # CLASSIFY SCHOOL EDUCATION TYPE
    # ==============================================================================

    def get_education_type(age):

        if age <= kindergarten_age:
            return "kindergarten"

        elif age <= elementary_age:
            return "elementary"

        elif age <= highschool_age:
            return "highschool"

        return None


    # ==============================================================================
    # SCHOOLS: ASSIGNED STUDENTS
    # ==============================================================================

    gdf_school_students = (
        gdf_activities[gdf_activities["purpose"] == "education"]
        .merge(
            df_persons[["person_id", "age", "household_id"]],
            on="person_id",
            how="inner",
            suffixes=("", "_person")
        )
    )

    gdf_school_students["education_type"] = (gdf_school_students["age"].apply(get_education_type))

    gdf_school_students = gdf_school_students[gdf_school_students["education_type"].notna()].copy()


    # ==============================================================================
    # IDENTIFY SCHOOLS
    #
    # A school is identified by:
    #
    #     geometry + education_type
    #
    # This is important because, for example, a kindergarten and an elementary
    # school can have exactly the same geometry.
    # ==============================================================================

    gdf_school_students["location_key"] = (
        gdf_school_students.geometry.apply(
            lambda x: x.wkb
        )
    )

    gdf_schools = (
        gdf_school_students[
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
        crs=gdf_school_students.crs
    )

    gdf_school_students = gdf_school_students.merge(
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


    # ==============================================================================
    # STUDENT HOMES
    # ==============================================================================

    gdf_school_students = gdf_school_students.merge(
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

    gdf_school_students = gpd.GeoDataFrame(
        gdf_school_students,
        geometry="home_geometry",
        crs=gdf_homes.crs
    )


    # ==============================================================================
    # SCHOOL CATCHMENT CIRCLES
    # ==============================================================================

    distance_crs = gdf_homes.estimate_utm_crs()

    gdf_schools_metric = gdf_schools.to_crs(
        distance_crs
    )

    gdf_school_students_metric = (
        gdf_school_students.to_crs(
            distance_crs
        )
    )

    school_circles = []

    for school_id in gdf_schools_metric["school_id"]:

        school = (
            gdf_schools_metric[
                gdf_schools_metric["school_id"] == school_id
            ]
            .iloc[0]
        )

        students = (
            gdf_school_students_metric[
                gdf_school_students_metric["school_id"] == school_id
            ]
            .copy()
        )

        students = students[
            students["home_geometry"].notna()
        ]

        if len(students) == 0:

            radius = 0

        else:

            distances = (
                students["home_geometry"]
                .distance(school.geometry)
            )

            radius = distances.quantile(
                0.90
            )

        school_circles.append(
            {
                "school_id": school_id,
                "education_type": school["education_type"],
                "n_students": len(students),
                "radius_m": radius,
                "geometry": school.geometry.buffer(radius)
            }
        )

    gdf_school_circles = gpd.GeoDataFrame(
        school_circles,
        crs=distance_crs
    )


    # ==============================================================================
    # GRAPH 1 — SCHOOL CATCHMENT AREAS
    #
    # No school labels because schools are dense.
    # ==============================================================================

    gdf_zones_metric = df_zones.to_crs(
        distance_crs
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    # Macrozone borders
    gdf_zones_metric.boundary.plot(
        ax=ax,
        linewidth=0.5,
        color="black"
    )

    # School catchment circles
    gdf_school_circles.plot(
        ax=ax,
        facecolor="none",
        edgecolor="red",
        linewidth=1.2
    )

    # School locations
    gdf_schools_metric.plot(
        ax=ax,
        markersize=20,
        color="black"
    )

    ax.set_title(
        "School Catchment Areas\n"
        "90th Percentile of Home-to-School Distance"
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        f"{analysis_path}/schools_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ==============================================================================
    # EXPECTED SCHOOL STUDENTS
    # ==============================================================================

    df_expected_schools = context.stage(
        "seville.data.education.merged"
    ).copy()

    df_expected_schools = df_expected_schools[(df_expected_schools["education_type"]!="university") &(df_expected_schools["education_type"]!="other")]

    # ==============================================================================
    # EXPECTED SCHOOL DATA
    #
    # Expected data uses:
    #
    #     geometry
    #     education_type
    #     students
    #
    # Multiple rows at the same geometry and education_type are aggregated.
    # ==============================================================================

    df_expected_schools = df_expected_schools.to_crs(
        gdf_schools.crs
    )

    df_expected_schools["location_key"] = (
        df_expected_schools.geometry.apply(
            lambda x: x.wkb
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


    # ==============================================================================
    # ASSIGNED STUDENTS PER SCHOOL
    # ==============================================================================

    assigned_school_counts = (
        gdf_school_students
        .groupby(
            [
                "location_key",
                "education_type",
                "school_id"
            ]
        )
        .size()
        .reset_index(
            name="assigned_students"
        )
    )


    # ==============================================================================
    # COMBINE EXPECTED + ASSIGNED
    # ==============================================================================

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


    # ==============================================================================
    # SCHOOL EXPECTED / ASSIGNED RATIOS
    # ==============================================================================

    total_expected_school_students = (
        school_stats["expected_students"].sum()
    )

    total_assigned_school_students = (
        school_stats["assigned_students"].sum()
    )

    if total_expected_school_students > 0:

        school_stats["expected_ratio"] = (
            school_stats["expected_students"]
            / total_expected_school_students
            * 100
        )

    else:

        school_stats["expected_ratio"] = 0


    if total_assigned_school_students > 0:

        school_stats["assigned_ratio"] = (
            school_stats["assigned_students"]
            / total_assigned_school_students
            * 100
        )

    else:

        school_stats["assigned_ratio"] = 0


    # ==============================================================================
    # GRAPH 2 — SCHOOL EXPECTED VS ASSIGNED RATIO
    #
    # Scatter plot instead of bars because there can be many schools.
    #
    # X = expected share of all school students
    # Y = assigned share of all school students
    #
    # The diagonal line represents:
    #
    #     assigned ratio = expected ratio
    #
    # Points above the line received more students than expected.
    # Points below the line received fewer students than expected.
    # ==============================================================================

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    max_ratio = max(
        school_stats["expected_ratio"].max(),
        school_stats["assigned_ratio"].max()
    )

    # 45-degree reference line
    ax.plot(
        [0, max_ratio],
        [0, max_ratio],
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.6,
        label="Expected = Assigned"
    )

    # Plot one point per education type
    education_types = (
        school_stats["education_type"]
        .dropna()
        .unique()
    )

    for education_type in education_types:

        subset = school_stats[
            school_stats["education_type"] == education_type
        ]

        ax.scatter(
            subset["expected_ratio"],
            subset["assigned_ratio"],
            s=45,
            alpha=0.7,
            label=education_type.capitalize()
        )

    ax.set_xlabel(
        "Expected student ratio (%)"
    )

    ax.set_ylabel(
        "Assigned student ratio (%)"
    )

    ax.set_title(
        "School Expected vs Assigned Student Ratios"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="Education type"
    )

    ax.set_xlim(
        0,
        max_ratio * 1.05
    )

    ax.set_ylim(
        0,
        max_ratio * 1.05
    )

    plt.tight_layout()

    plt.savefig(
        f"{analysis_path}/school_expected_vs_assigned.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ==============================================================================
    # UNIVERSITIES: STUDENTS ABOVE HIGHSCHOOL GRADUATION AGE
    # ==============================================================================

    gdf_university_students = (
        gdf_activities[
            gdf_activities["purpose"] == "education"
        ]
        .merge(
            df_persons[
                [
                    "person_id",
                    "age"
                ]
            ],
            on="person_id",
            how="inner",
            suffixes=("", "_person")
        )
    )

    gdf_university_students = (
        gdf_university_students[
            gdf_university_students["age"]
            > highschool_age
        ]
        .copy()
    )


    # ==============================================================================
    # IDENTIFY UNIVERSITIES
    # ==============================================================================

    gdf_university_students["location_key"] = (
        gdf_university_students.geometry.apply(
            lambda x: x.wkb
        )
    )

    gdf_universities = (
        gdf_university_students
        .drop_duplicates(
            "location_key"
        )
        .drop(
            columns="location_key"
        )
        .copy()
    )

    gdf_universities = gpd.GeoDataFrame(
        gdf_universities,
        geometry="geometry",
        crs=gdf_university_students.crs
    )


    # ==============================================================================
    # UNIVERSITY EXPECTED DATA
    # ==============================================================================

    uni_locations = (
        context.stage(
            "seville.data.education.universities"
        )
        .copy()
    )

    uni_locations = uni_locations.to_crs(
        gdf_universities.crs
    )

    uni_locations["expected_ratio"] = (
        uni_locations["students"]
        / uni_locations["students"].sum()
        * 100
    )

    gdf_universities = gdf_universities.merge(
        uni_locations[
            [
                "geometry",
                "location_name",
                "expected_ratio"
            ]
        ],
        on="geometry",
        how="left"
    )

    gdf_universities["university_id"] = (
        gdf_universities["location_name"]
    )


    # ==============================================================================
    # COUNT ASSIGNED UNIVERSITY STUDENTS
    # ==============================================================================

    university_counts = (
        gdf_university_students
        .groupby(
            "geometry"
        )
        .size()
        .reset_index(
            name="assigned_students"
        )
    )

    university_counts = gpd.GeoDataFrame(
        university_counts,
        geometry="geometry",
        crs=gdf_university_students.crs
    )


    # ==============================================================================
    # ATTACH UNIVERSITY IDS + EXPECTED RATIOS
    # ==============================================================================

    university_counts = university_counts.merge(
        gdf_universities[
            [
                "geometry",
                "university_id",
                "expected_ratio"
            ]
        ],
        on="geometry",
        how="left"
    )


    # ==============================================================================
    # UNIVERSITY ASSIGNED RATIO
    # ==============================================================================

    total_university_students = (
        university_counts["assigned_students"].sum()
    )

    if total_university_students > 0:

        university_counts["assigned_ratio"] = (
            university_counts["assigned_students"]
            / total_university_students
            * 100
        )

    else:

        university_counts["assigned_ratio"] = 0


    # Sort by expected ratio
    university_counts = university_counts.sort_values(
        "expected_ratio",
        ascending=False
    )


    # ==============================================================================
    # GRAPH 3 — UNIVERSITY EXPECTED VS ASSIGNED RATIO
    # ==============================================================================

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    x = np.arange(
        len(university_counts)
    )

    width = 0.38

    ax.bar(
        x - width / 2,
        university_counts["expected_ratio"],
        width,
        label="Expected ratio"
    )

    ax.bar(
        x + width / 2,
        university_counts["assigned_ratio"],
        width,
        label="Assigned ratio"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        university_counts["university_id"],
        rotation=45,
        ha="right"
    )

    ax.set_ylabel(
        "Share of students (%)"
    )

    ax.set_xlabel(
        "University"
    )

    ax.set_title(
        "Expected vs Assigned University Student Ratio"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        f"{analysis_path}/university_expected_vs_assigned.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ==============================================================================
    # GRAPH 4 — UNIVERSITY LOCATIONS
    # ==============================================================================

    gdf_universities_plot = (
        gdf_universities.to_crs(
            df_zones.crs
        )
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    # Background
    df_zones.plot(
        ax=ax,
        facecolor="white",
        edgecolor="black",
        linewidth=0.7
    )

    # University locations
    gdf_universities_plot.plot(
        ax=ax,
        color="red",
        edgecolor="black",
        markersize=100,
        zorder=5
    )

    # University labels
    for _, university in (
        gdf_universities_plot.iterrows()
    ):

        x = university.geometry.x
        y = university.geometry.y

        ax.text(
            x,
            y,
            f"{university['university_id']}",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=6
        )

    ax.set_title(
        "University Locations"
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        f"{analysis_path}/university_locations.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ==============================================================================
    # SAVE SUMMARY TABLES
    # ==============================================================================

    school_stats.drop(
        columns=[
            "location_key"
        ],
        errors="ignore"
    ).to_csv(
        f"{analysis_path}/school_expected_vs_assigned.csv",
        index=False
    )

    university_counts.drop(
        columns=[
            "geometry"
        ],
        errors="ignore"
    ).to_csv(
        f"{analysis_path}/university_expected_vs_assigned.csv",
        index=False
    )


    # ==============================================================================
    # PRINT SUMMARY
    # ==============================================================================

    print(
        "\nSchool expected vs assigned:"
    )

    print(
        school_stats[
            [
                "school_id",
                "education_type",
                "expected_students",
                "assigned_students",
                "expected_ratio",
                "assigned_ratio"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nUniversity expected vs assigned:")

    print(
        university_counts[
            [
                "university_id",
                "assigned_students",
                "expected_ratio",
                "assigned_ratio"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nEducation analysis completed.")

    print(f"Outputs written to: {analysis_path}")

class DebugContext:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.snapshot = pickle.load(f)

    def config(self, name, *args, **kwargs):
        return self.snapshot["config"][name]

    def stage(self, name, *args, **kwargs):
        return self.snapshot["stage"][name]

if __name__ == "__main__":
    context = DebugContext("../export/snapshot.pkl")
    print(context.snapshot["config"])
    execute(context)
