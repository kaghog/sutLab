# Generating the Hannover population

The following sections describe how to generate a synthetic population for
Hannover using the pipeline. First all necessary data must be gathered.
Afterwards, the pipeline can be run to create a synthetic population in *CSV*
and *GPKG* format. These outputs can be used for analysis, or serve as input
to a MATSim simulation.

This guide will cover the following steps:

- [Gathering the data](#section-data)
- [Running the pipeline](#section-population)
- [Running a simulation](#section-simulation)

## <a name="section-data"></a>Gathering the data

To create the scenario, a couple of data sources must be collected. It is best
to start with an empty folder, e.g. `/data`. All data sets need to be named
in a specific way and put into specific sub-directories. The following paragraphs
describe this process.

### 1) Hannover administrative units

- Use Shapefile of *SKH20_Mikrobezirke* which contains Mikrobezirke levels administrative units (390 areas that are similar in terms of population number)
- Put the following files into `/data/hannover`:
  - SKH20_Mikrobezirke.cpg
  - SKH20_Mikrobezirke.dbf
  - SKH20_Mikrobezirke.prj
  - SKH20_Mikrobezirke.shp
  - SKH20_Mikrobezirke.shx

### 2) German driving license ownership data (municipality; Germany sex, age, type; Bundesland, sex, type)

- [License ownership data](https://www.kba.de/DE/Statistik/Kraftfahrer/Fahrerlaubnisse/Fahrerlaubnisbestand/fahrerlaubnisbestand_node.html)
- Select *2024* from the dropdown list and click *Auswahl anwenden*
- Download the *XLSX* file at the bottom of the page
- Put the resulting file into `/data/germany`:
  - fe4_2024.xlsx


### 3) Hannover buildings

- Use Shapefile *buildings_Hannover_20km* which contains the space covered by buildings in Hannover + 20km buffer
- Put the following files into `/data/hannover/buildings`:
  - buildings_Hannover_20km.cpg
  - buildings_Hannover_20km.dbf
  - buildings_Hannover_20km.prj
  - buildings_Hannover_20km.shp
  - buildings_Hannover_20km.shx

### 4) German National household travel survey (MiD 2017)

- Use the german national household travel survey *MiD2017_B1_Datensatzpaket*
- Put the following *csv* files in to the folder `data/mid2017`:
  - MiD2017_Haushalte.csv
  - MiD2017_Personen.csv
  - MiD2017_Wege.csv

### 5) OpenStreetMap (Hannover)

- [Geofabrik Niedersachsen](https://download.geofabrik.de/europe/germany/niedersachsen.html)
- Download the data set for **Niedersachsen** in *osm.pbf* format ("Commonly used formats"), which is state of Hannover
- Put the resulting *osm.pbf* file into `/data/osm`: 
  - niedersachsen-latest.osm.pbf

### Overview

Your folder structure should now have at least the following files:

- `data/hannover/SKH20_Mikrobezirke.cpg`
- `data/hannover/SKH20_Mikrobezirke.dbf`
- `data/hannover/SKH20_Mikrobezirke.prj`
- `data/hannover/SKH20_Mikrobezirke.shp`
- `data/hannover/SKH20_Mikrobezirke.shx`
- `data/hannover/buildings/buildings_Hannover_20km.cpg`
- `data/hannover/buildings/buildings_Hannover_20km.dbf`
- `data/hannover/buildings/buildings_Hannover_20km.prj`
- `data/hannover/buildings/buildings_Hannover_20km.shp`
- `data/hannover/buildings/buildings_Hannover_20km.shx`
- `data/mid_2017/MiD2017_Haushalte.csv`
- `data/mid_2017/MiD2017_Personen.csv`
- `data/mid_2017/MiD2017_Wege.csv`
- `data/germany/fe4_2024.xlsx`
- `data/osm/niedersachsen-latest.osm.pbf`

## Preparing the environment

The tool `osmconvert` must be accessible from the command line for the pipeline. To do so, it can be located next to the code or inserted into the PATH variable of Linux/Windows. Precompiled binaries can be downloaded on [this website](https://wiki.openstreetmap.org/wiki/Osmconvert).

## <a name="section-population">Running the pipeline

The pipeline code is available in [this repository](https://github.com/eqasim-org/hannover.git).
To use the code, you have to clone the repository with `git`:

```bash
git clone https://github.com/eqasim-org/hannover.git pipeline
```

which will create the `pipeline` folder containing the pipeline code. To
set up all dependencies, especially the [synpp](https://github.com/eqasim-org/synpp) package,
which is the code of the pipeline code, we recommend setting up a Python
environment using [Anaconda](https://www.anaconda.com/):

```bash
cd pipeline
conda env create -f environment.yml -n hannover
```

This will create a new Anaconda environment with the name `hannover`.

To activate the environment, run:

```bash
conda activate hannover
```

Now have a look at `config_hannover.yml` which is the configuration of the pipeline code.
Have a look at [synpp](https://github.com/eqasim-org/synpp) in case you want to get a more general
understanding of what it does. For the moment, it is important to adjust
two configuration values inside of `config_hannover.yml`:

- `working_directory`: This should be an *existing* (ideally empty) folder where
the pipeline will put temporary and cached files during runtime.
- `data_path`: This should be the path to the folder where you were collecting
and arranging all the raw data sets as described above.
- `output_path`: This should be the path to the folder where the output data
of the pipeline should be stored. It must exist and should ideally be empty
for now.

To set up the working/output directory, create, for instance, a `cache` and a
`output` directory. These are already configured in `config.yml`:

```bash
mkdir cache
mkdir output
```

Everything is set now to run the pipeline. The way `config_hannover.yml` is configured
it will create the relevant output files in the `output` folder.

To run the pipeline, call the [synpp](https://github.com/eqasim-org/synpp) runner:

```bash
python3 -m synpp config_hannover.yml
```

It will read `config_hannover.yml`, process all the pipeline code
and eventually create the synthetic population. You should see a couple of
stages running one after another. Most notably, first, the pipeline will read all
the raw data sets to filter them and put them into the correct internal formats.

After running, you should be able to see a couple of files in the `output`
folder:

- `meta.json` contains some meta data, e.g. with which random seed or sampling
rate the population was created and when.
- `persons.csv` and `households.csv` contain all persons and households in the
population with their respective sociodemographic attributes.
- `activities.csv` and `trips.csv` contain all activities and trips in the
daily mobility patterns of these people including attributes on the purposes
of activities.
- `activities.gpkg` and `trips.gpkg` represent the same trips and
activities, but in the spatial *GPKG* format. Activities contain point
geometries to indicate where they happen and the trips file contains line
geometries to indicate origin and destination of each trip.

## Generating smaller areas

You can create smaller areas in Bavaria by selecting the *Regierungsbezirk* that you would like to generate. For that, activate the following option in the configuration:

```yaml
hannover.political_prefix: ["091", "092", "097"] # Oberbayern, Niederbayern, Schwaben
```

In that case, only the *Bezirke* with the identifiers *091*, *092*, and *093* are generated. Those are the ones that are directly located around hannover.
