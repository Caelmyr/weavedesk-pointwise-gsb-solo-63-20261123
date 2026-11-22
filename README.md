# WeaveDesk

WeaveDesk is a local image workbench for inspecting mounted textiles. Register repeat photographs to the same physical plane, read yarn direction and surface appearance, and retain measurements beside original capture evidence.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19980`. Set `WEAVEDESK_PORT` to use a different local port. A named volume holds images and measurement history. With Python 3.11 and the pinned packages installed, run `./start.sh`. Set `WEAVEDESK_DEPS_IMAGE` to a dependency-stage image with the locked packages if building offline.

## Examine a mounted textile

1. Load the four marked synthetic textiles or import the substrate and repeat captures. In **Captures**, enter the physical sheet dimensions, set the four surrounding registration marks, and register the other photographs. Compare the registered views and inspect their support.
2. In **Sampling regions**, outline an appropriate patch. **Yarn spacing and angle** measures two approximately perpendicular direction sectors and saves a spectral estimate for each supported yarn family. Mark the approximate first axis when the fabric has visible skew. A missing peak leaves its row blank. Choose **Yarn knot census** for isolated dark details, their areas and positions.
3. The workbench also measures color variation, coverage against marked references, local spots, polarized glare, raking relief, texture repeats, line scans, edge color, fine-line networks, binding width, edge stitch visibility, mounting seams, and pale fiber flecks. Use Geometry to save lengths and areas. Approve a reference patch or compare saved measurements by mounted group.
4. Pin an Evidence report with selected measurements and optional original captures. The local received-evidence viewer checks declared members in an imported ZIP.

## Mount a textile in a clamp frame

1. With the scaled front photograph registered and the specimen dimensions entered, circle the edge that keeps shedding as a **Thread-loss risk area**, the woven label viewers must see as **Keep visible for viewers**, and any missing fabric as **Missing fabric (loss)**. Mark the ruler and other non-fabric as **Exclude from measurement**.
2. In **Mounting**, enter the ready-made frame: bar contact width, center-to-center bar spacing, and the detent positions of the linked bars. The workbench evaluates each detent — both bars must land on the plane and press enough surviving fabric, since losses never count as pressed — then adopts the detent that holds the most at-risk cloth while keeping the label exposed. The plan shows where the two bars actually land and which stretches of the risk area remain unheld.
3. After the frame comes off, re-photograph at the same scale and position, register the capture, and run **Surface observations** on it. **Review after removal** compares those with the earlier spot observations at the adopted bars' positions: marks already present are not reported as new, only changes clearly visible in both sessions and inside the pressed bands enter the confirmable changes, and obscured stretches are listed for re-photography. The review record links back to the adopted landing positions of the clamp frame.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. A specimen holds up to ten captures and 24 regions. The registered view is limited to 1,800 pixels on its longest side. Yarn spacing estimates from visible texture are approximate and do not identify individual thread ancestry. The knot tool does not decompose joined knots.

The local PressLoupe workbench supplied image registration, color and optical inspection, and evidence archive handling. WeaveDesk adds a synthetic interlaced textile series and a separate physical yarn-spacing calculation integrated with the measurement table, image overlay, CSV and saved results.
