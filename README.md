# WeaveDesk

WeaveDesk is a local image workbench for inspecting mounted textiles. Register repeat photographs to the same physical plane, read yarn direction and surface appearance, and retain measurements beside original capture evidence.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19980`. Set `WEAVEDESK_PORT` to use a different local port. A named volume holds images and measurement history. With Python 3.11 and the pinned packages installed, run `./start.sh`. Set `WEAVEDESK_DEPS_IMAGE` to a dependency-stage image with the locked packages if building offline.

## Examine a mounted textile

1. Load the four marked synthetic textiles or import the substrate and repeat captures. In **Captures**, enter the physical sheet dimensions, set the four surrounding registration marks, and register the other photographs. Compare the registered views and inspect their support.
2. In **Sampling regions**, outline an appropriate patch. **Yarn spacing and angle** measures two approximately perpendicular direction sectors and saves a spectral estimate for each supported yarn family. Mark the approximate first axis when the fabric has visible skew. A missing peak leaves its row blank. Choose **Yarn knot census** for isolated dark details, their areas and positions.
3. The workbench also measures color variation, coverage against marked references, local spots, polarized glare, raking relief, texture repeats, line scans, edge color, fine-line networks, binding width, edge stitch visibility, mounting seams, and pale fiber flecks. Use Geometry to save lengths and areas. Approve a reference patch or compare saved measurements by mounted group.
   **Polarized glare** keeps one adopted glare line per specimen. Mark one confirmed film-glare patch and one confirmed bright-yarn patch, then adopt the line once; every later polarized run and reference re-check on that specimen reads the same line. The map shows where color is retained, where the light makes it unreliable, where there is too little detail, and where nothing is visible.
   **Reference match** on a specimen with an adopted line needs a fresh parallel/cross pair. It reports color difference only where both the approved sample and the new photo are visible and admitted by the same line. Positions the old photo left blank, or that the light makes unreliable, stay marked and never become ordinary color differences.
4. Pin an Evidence report with selected measurements and optional original captures. The local received-evidence viewer checks declared members in an imported ZIP.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. A specimen holds up to ten captures and 24 regions. The registered view is limited to 1,800 pixels on its longest side. Yarn spacing estimates from visible texture are approximate and do not identify individual thread ancestry. The knot tool does not decompose joined knots.

The local PressLoupe workbench supplied image registration, color and optical inspection, and evidence archive handling. WeaveDesk adds a synthetic interlaced textile series and a separate physical yarn-spacing calculation integrated with the measurement table, image overlay, CSV and saved results.
