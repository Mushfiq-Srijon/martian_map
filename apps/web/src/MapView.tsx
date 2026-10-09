import { useEffect, useRef, useState } from 'react';
import 'ol/ol.css';
import OLMap from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import XYZ from 'ol/source/XYZ';
import TileGrid from 'ol/tilegrid/TileGrid';
import Projection from 'ol/proj/Projection';

const MARS_RADIUS_M = 3396190;
const TILE_BASE_URL = '/tiles';

type Level = {
  columns: number;
  rows: number;
  metres_per_pixel: number;
};

type Manifest = {
  name: string;
  source: string;
  generated_utc: string;
  tile_size: number;
  min_zoom: number;
  max_zoom: number;
  bounds_m: { left: number; bottom: number; right: number; top: number };
  levels: Record<string, Level>;
};

export default function MapView() {
  const mapDivRef = useRef<HTMLDivElement>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cursor, setCursor] = useState<string>('');

  // Load the tile manifest written by tile_dem.py.
  useEffect(() => {
    let cancelled = false;
    fetch(`${TILE_BASE_URL}/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`manifest request failed (HTTP ${r.status})`);
        return r.json() as Promise<Manifest>;
      })
      .then((m) => {
        if (!cancelled) setManifest(m);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Build the map once the manifest is available.
  useEffect(() => {
    const div = mapDivRef.current;
    if (!manifest || !div) return;

    const { bounds_m: b, levels, tile_size, max_zoom } = manifest;
    const extent = [b.left, b.bottom, b.right, b.top];

    // Mars is not Web Mercator, so we declare a custom projection in metres.
    const projection = new Projection({
      code: 'MARS:EQC',
      units: 'm',
      extent,
    });

    // One resolution per zoom level, in metres per screen pixel.
    // Index 0 is zoom 0, and so on. The manifest stores these per level.
    const resolutions: number[] = [];
    for (let z = 0; z <= max_zoom; z++) {
      resolutions.push(levels[String(z)].metres_per_pixel);
    }

    // Tile grid with the origin at the top-left, matching the y=0-at-north
    // layout written by tile_dem.py.
    const tileGrid = new TileGrid({
      extent,
      origin: [b.left, b.top],
      resolutions,
      tileSize: tile_size,
    });

    const source = new XYZ({
      url: `${TILE_BASE_URL}/{z}/{x}/{y}.png`,
      projection,
      tileGrid,
      wrapX: false,
      attributions: `${manifest.source}. Shaded elevation rendered from NASA data.`,
    });

    const view = new View({
      projection,
      resolutions,
      extent,
      center: [0, 0],
      resolution: resolutions[0],
      enableRotation: false,
    });

    const map = new OLMap({
      target: div,
      layers: [new TileLayer({ source })],
      view,
    });

    // Fit the whole planet into the container on load.
    view.fit(extent, { size: [div.clientWidth, div.clientHeight] });

    // Show latitude and longitude under the cursor.
    map.on('pointermove', (evt) => {
      const [x, y] = evt.coordinate;
      const lonDeg = (x / MARS_RADIUS_M) * (180 / Math.PI);
      const latDeg = (y / MARS_RADIUS_M) * (180 / Math.PI);
      setCursor(`${latDeg.toFixed(2)}°, ${lonDeg.toFixed(2)}°`);
    });

    return () => {
      map.setTarget(undefined);
    };
  }, [manifest]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh' }}>
      <div
        ref={mapDivRef}
        style={{ width: '100%', height: '100%', background: '#1a1220' }}
      />
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          padding: '8px 12px',
          borderRadius: 8,
          background: 'rgba(20, 14, 24, 0.82)',
          color: '#f2e6dc',
          fontFamily: 'system-ui, sans-serif',
          fontSize: 13,
          pointerEvents: 'none',
        }}
      >
        <strong>{manifest?.name ?? 'Mars map'}</strong>
        <div>{manifest ? `Source: ${manifest.source}` : 'Loading map...'}</div>
        <div>Cursor: {cursor || '—'}</div>
        {error && <div style={{ color: '#ff9b8a' }}>Error: {error}</div>}
      </div>
    </div>
  );
}