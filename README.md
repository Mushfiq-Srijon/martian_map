# Martian Map

Layered, integrated view of Martian routes and sites, built from NASA mission data
for planning a safe and productive Marswalk.

## Structure

- `apps/web`: React + TypeScript frontend
- `apps/api`: NestJS backend (added later)
- `pipeline`: Python data processing (DEM tiling, hazard layers, routing cost surfaces)
- `data`: local raw and processed data (not committed)

## Data sources

- MOLA gridded elevation (Mars Global Surveyor), via USGS Astropedia
- HiRISE and CTX imagery (Mars Reconnaissance Orbiter), via the PDS Imaging Node
- THEMIS thermal inertia (Mars Odyssey)
- Mars Rover Photos API (api.nasa.gov)

All values shown in the app are labeled with their source and date.