import numpy as np

"""
This stage updates the formatting of the population and employment census data sets such
that they can be procesed by the IPF algorithm.
"""


def configure(context):
    context.stage("hannover.data.census.population")
    context.stage("hannover.data.census.employment")
    context.stage("hannover.data.census.licenses")


def execute(context):
    # Load data
    df_population = context.stage("hannover.data.census.population")
    df_employment = context.stage("hannover.data.census.employment")

    df_licenses_country = context.stage("hannover.data.census.licenses")[0]
    df_licenses_kreis = context.stage("hannover.data.census.licenses")[2]

    # Generate numeric sex
    df_population["sex"] = df_population["sex"].replace({"male": 1, "female": 2})
    df_employment["sex"] = df_employment["sex"].replace({"male": 1, "female": 2})
    df_licenses_country["sex"] = df_licenses_country["sex"].replace(
        {"male": 1, "female": 2}
    )

    # Validation
    unique_population_kreis = set(df_population["kreis_code"].unique())
    unique_employment_kreis = set(df_employment["kreis_code"].unique())
    unique_licenses_kreis = set(df_licenses_kreis["kreis_code"].unique())
    assert unique_population_kreis == unique_employment_kreis, (
        f"Population kreis {unique_population_kreis} != Employment kreis {unique_employment_kreis}"
    )
    assert unique_population_kreis == unique_licenses_kreis, (
        f"Population kreis {unique_population_kreis} != Licenses kreis {unique_licenses_kreis}"
    )

    # Create indices for IPF (at kreis level and commune level)
    unique_communes = np.sort(df_population["commune_id"].unique())
    unique_kreis = np.sort(
        list(
            set(df_employment["kreis_code"].unique())
            | set(df_population["kreis_code"].unique())
        )
    )
    unique_departements = np.sort(df_population["departement_id"].unique())

    commune_mapping = {c: k for k, c in enumerate(unique_communes)}
    kreis_mapping = {c: k for k, c in enumerate(unique_kreis)}
    departement_mapping = {c: k for k, c in enumerate(unique_departements)}

    df_population["commune_index"] = df_population["commune_id"].replace(
        commune_mapping
    )
    df_population["kreis_index"] = df_population["kreis_code"].replace(kreis_mapping)
    df_population["departement_index"] = df_population["departement_id"].replace(
        departement_mapping
    )
    df_employment["kreis_index"] = df_employment["kreis_code"].replace(kreis_mapping)
    df_licenses_kreis["kreis_index"] = df_licenses_kreis["kreis_code"].replace(
        kreis_mapping
    )

    ## Licenses

    # Consolidate municipalities at kreis level
    for kreis_code in df_licenses_kreis["kreis_code"].unique():
        population = df_population.loc[
            df_population["kreis_code"] == kreis_code, "weight"
        ].sum()
        licenses = df_licenses_kreis.loc[
            df_licenses_kreis["kreis_code"] == kreis_code, "weight"
        ].sum()

        if licenses > population:
            factor = population / licenses
            df_licenses_kreis.loc[
                df_licenses_kreis["kreis_code"] == kreis_code, "weight"
            ] *= factor
            print("Adapting licenses for {} by factor {}".format(kreis_code, factor))

    # Scale up the sociodemographics for the study area
    df_licenses_country["weight"] = (
        df_licenses_country["relative_weight"] * df_licenses_kreis["weight"].sum()
    )

    # Consolidate sex and age
    population_age_classes = np.sort(df_population["age_class"].unique())
    license_age_classes = np.sort(df_licenses_country["age_class"].unique())

    joint_age_classes = np.sort(
        list(set(population_age_classes) & set(license_age_classes))
    )
    joint_age_upper = list(joint_age_classes[1:]) + [9999]

    for sex in [1, 2]:
        for lower, upper in zip(joint_age_classes, joint_age_upper):
            f_population = df_population["sex"] == sex
            f_population &= df_population["age_class"] >= lower
            f_population &= df_population["age_class"] < upper

            f_license = df_licenses_country["sex"] == sex
            f_license &= df_licenses_country["age_class"] >= lower
            f_license &= df_licenses_country["age_class"] < upper

            population = df_population.loc[f_population, "weight"].sum()
            licenses = df_licenses_country.loc[f_license, "weight"].sum()

            if population < licenses:
                factor = population / licenses

                print(
                    "Adapting sex:{} age:({}, {}) by factor {}".format(
                        ["", "m", "f"][sex], lower, upper, factor
                    )
                )

                df_licenses_country.loc[f_license, "weight"] *= factor

    # Take into account updated total
    factor = df_licenses_country["weight"].sum() / df_licenses_kreis["weight"].sum()
    print("Adapting total with correction factor {}".format(factor))
    df_licenses_kreis["weight"] *= factor

    return df_population, df_employment, df_licenses_country, df_licenses_kreis
