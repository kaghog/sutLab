#!/bin/bash
# Clear cache for secondary location assignment and dependent analysis stages

echo "Clearing secondary locations cache..."
rm -rf cache/synthesis.population.spatial.secondary.locations__*.cache
rm -rf cache/synthesis.population.spatial.secondary.locations__*.p
rm -rf cache/synthesis.population.spatial.locations__*.cache
rm -rf cache/synthesis.population.spatial.locations__*.p

echo "Clearing dependent analysis cache..."
rm -rf cache/analysis.hannover.ivt_style.analysis__*.cache
rm -rf cache/analysis.hannover.ivt_style.analysis__*.p
rm -rf cache/analysis.synthesis.mode_distances__*.cache
rm -rf cache/analysis.synthesis.mode_distances__*.p

