# WeaveDesk

WeaveDesk is a local image workbench for inspecting mounted textiles. Register repeat photographs to the same physical plane, read yarn direction and surface appearance, and retain measurements beside original capture evidence.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19980`. Set `WEAVEDESK_PORT` to use a different local port. A named volume holds images and measurement history. With Python 3.11 and the pinned packages installed, run `./start.sh`. Set `WEAVEDESK_DEPS_IMAGE` to a dependency-stage image with the locked packages if building offline.

## Examine a mounted textile

1. Load the four marked synthetic textiles or import the substrate and repeat captures. In **Captures**, enter the physical sheet dimensions, set the four surrounding registration marks, and register the other photographs. Compare the registered views and inspect their support.
2. In **Sampling regions**, outline an appropriate patch. **Yarn spacing and angle** measures two approximately perpendicular direction sectors and saves a spectral estimate for each supported yarn family. Mark the approximate first axis when the fabric has visible skew. A missing peak leaves its row blank. Choose **Yarn knot census** for isolated dark details, their areas and positions.
3. The workbench also measures color variation, coverage against marked references, local spots, polarized glare, raking relief, texture repeats, line scans, edge color, fine-line networks, binding width, edge stitch visibility, mounting seams, and pale fiber flecks. Use Geometry to save lengths and areas. Approve a reference patch or compare saved measurements by mounted group.
4. For a raised embroidery dust cover, run **Pile relief height** on the two registered raking-light captures together with the diffuse front photo. Circle the cast shadows of two hand-measured piles (for example the easily abraded edge cluster and the tallest middle cluster); the workbench keeps one workshop-wide shadow-to-height ruler, calibrated on the edge anchor by default, and reports the second anchor as a cross-check with its error. A cluster only receives a height when its shadow is confirmed in *both* raking photos; dark base dye visible in the diffuse photo is excluded and never counted as a shadow. Then draw a **Sloped cover edge** in Geometry across the yarns: it returns the real incline length over measured piles, the flat front-photo projection as a reference, and explicit unestimated gap lengths that are not flattened into the cover total.
4. Pin an Evidence report with selected measurements and optional original captures. The local received-evidence viewer checks declared members in an imported ZIP.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. A specimen holds up to ten captures and 24 regions. The registered view is limited to 1,800 pixels on its longest side. Yarn spacing estimates from visible texture are approximate and do not identify individual thread ancestry. The knot tool does not decompose joined knots.

The local PressLoupe workbench supplied image registration, color and optical inspection, and evidence archive handling. WeaveDesk adds a synthetic interlaced textile series and a separate physical yarn-spacing calculation integrated with the measurement table, image overlay, CSV and saved results.
