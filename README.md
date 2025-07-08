# sutLab

## Hannover Synthetic Population

This part contains the synthetic population generation pipeline for Hannover, including all scripts, configurations, and analysis tools.

The pipeline generates:

- Synthetic households and persons with sociodemographic attributes
- Daily activity schedules and trips
- Outputs in both **CSV** and **GeoPackage (GPKG)** formats for direct use in **MATSim simulations** and related spatial analyses.

### Data preparation and usage

To run the pipeline successfully, you need to gather and prepare several input datasets, such as:

- Administrative boundaries of Hannover
- MiD 2017 German national travel survey
- Buildings
- Driving license statistics
- OpenStreetMap

For step-by-step instructions on gathering and structuring the necessary data, please refer to the [Data preparation](./docs/population.md).

To run the analysis, add the stage 'analysis.hannover.ivt_style.analysis'. This will produce plots for comparison between HTS and synthesis.
