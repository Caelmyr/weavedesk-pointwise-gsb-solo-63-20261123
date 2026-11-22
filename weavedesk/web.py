import json
import os
from pathlib import Path
from threading import BoundedSemaphore

from flask import Blueprint, Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from .analysis import approve_reference, evaluate
from .demo import load_training, preview_photo
from .evidence import (
    artifact_json,
    create_evidence,
    evidence_zip,
    filename_stem,
    result_metrics_table,
    specimen_html,
    study_table,
    summarize_evidence,
    summarize_result,
)
from .inspection import side_preview, viewing_pair
from .art import base_rgba
from .intake import MAX_ARCHIVE_BYTES, inspect_packet
from .metrology import create_metrology
from .model import (
    ANALYSIS_LABELS,
    MAX_UPLOAD,
    REGION_PURPOSES,
    ROLE_LABELS,
    ConflictError,
    InputError,
    MissingError,
    dictionary,
    enumeration,
    flag_record_fields,
    identity,
    match_slot,
    model_version,
    string,
    tile_fields,
)
from .parameters import TOOLS
from .quality import quality_manifest
from .photos import png_bytes
from .review import parse_reviews, review_selection, save_reviews
from .store import Store
from .study import create_study, public_metrics
from .survey import clear_survey_plan, save_survey_plan, survey_preview
from .workspace import (
    calibrate,
    clear_calibration,
    edit_photo,
    edit_specimen,
    import_photo,
    present_tile,
    preview_transform,
    remove_photo,
    remove_region,
    save_plane,
    save_region,
    save_registration,
    summarize_tile,
)


STATIC = Path(__file__).resolve().parents[1] / "static"


def error_response(error, status):
    return jsonify(error=str(error), status=status), status


def json_request():
    if not request.is_json:
        raise InputError("Send a JSON request body.")
    return dictionary(request.get_json(), "Request")


def binary_response(payload, mime, filename=None):
    response = Response(payload, mimetype=mime)
    if filename:
        response.headers.set("Content-Disposition", f'attachment; filename="{filename}"')
    return response


def image_response(image):
    return binary_response(png_bytes(image), "image/png")


def png_blob(store, digest):
    if digest is None:
        raise MissingError("No image exists for this selected layer.")
    return send_file(store.blobs.require(digest), mimetype="image/png", conditional=True, max_age=48 * 3600)


def register_specimens(api, store, heavy):
    @api.get("/specimens")
    def specimens():
        items = [summarize_tile(tile) for tile in store.list_tiles()]
        return jsonify({"items": items})

    @api.post("/specimens")
    def specimen_create():
        return jsonify(present_tile(store.create_tile(tile_fields(json_request())))), 201

    @api.get("/specimens/<specimen_id>")
    def specimen_show(specimen_id):
        return jsonify(present_tile(store.tile(specimen_id)))

    @api.patch("/specimens/<specimen_id>")
    def specimen_change(specimen_id):
        return jsonify(present_tile(edit_specimen(store, specimen_id, json_request())))

    @api.get("/specimens/<specimen_id>/activity")
    def specimen_timeline(specimen_id):
        return jsonify({"items": store.events(specimen_id)})

    @api.post("/specimens/<specimen_id>/photos")
    def photo_upload(specimen_id):
        capture = request.files.get("photo")
        if capture is None:
            raise InputError("Choose a photo to upload.")
        try:
            revision = int(request.form.get("version", ""))
        except ValueError as error:
            raise InputError("Reload the specimen before uploading photos.") from error
        role = request.form.get("role", "diffuse")
        note = request.form.get("note", "")
        file_content = capture.stream.read(MAX_UPLOAD + 1)
        with heavy:
            changed = import_photo(store, specimen_id, revision, file_content, capture.filename, role, note)
        return jsonify(present_tile(changed)), 201

    @api.patch("/specimens/<specimen_id>/photos/<photo_id>")
    def photo_update(specimen_id, photo_id):
        return jsonify(present_tile(edit_photo(store, specimen_id, photo_id, json_request())))

    @api.delete("/specimens/<specimen_id>/photos/<photo_id>")
    def photo_remove(specimen_id, photo_id):
        return jsonify(present_tile(remove_photo(store, specimen_id, photo_id, json_request())))

    @api.get("/specimens/<specimen_id>/photos/<photo_id>/<layer>")
    def photo_layer(specimen_id, photo_id, layer):
        tile = store.tile(specimen_id)
        photo = match_slot(tile["photos"], identity(photo_id, "Photo ID"), "photo")
        layers = {
            "preview": "preview_digest",
            "thumbnail": "thumbnail_digest",
            "original": "source_digest",
        }
        chosen = enumeration(layer, layers, "Photo layer")
        digest = photo[layers[chosen]]
        if chosen == "original":
            return send_file(
                store.blobs.require(digest),
                download_name=photo["name"],
                as_attachment=True,
                mimetype=photo["original_mime"],
            )
        return png_blob(store, digest)

    @api.put("/specimens/<specimen_id>/plane")
    def plane_save(specimen_id):
        return jsonify(present_tile(save_plane(store, specimen_id, json_request())))

    @api.put("/specimens/<specimen_id>/registrations/<photo_id>")
    def registration_save(specimen_id, photo_id):
        with heavy:
            specimen = save_registration(store, specimen_id, photo_id, json_request())
        return jsonify(present_tile(specimen))

    @api.post("/specimens/<specimen_id>/geometry/<photo_id>/preview")
    def geometry_test(specimen_id, photo_id):
        data = json_request()
        mode = enumeration(data.get("mode"), ("base", "registration"), "Preview mode")
        with heavy:
            frame = preview_transform(store, specimen_id, photo_id, data, mode)
        response = image_response(base_rgba(frame))
        response.headers["X-Specimen-Support"] = f"{frame.valid.mean() * 100:.2f}"
        return response

    @api.get("/specimens/<specimen_id>/viewer/<photo_id>")
    def viewer(specimen_id, photo_id):
        selected = enumeration(request.args.get("layer", "raw"), ("raw", "regions", "corrected"), "Viewer layer")
        tile = store.tile(specimen_id)
        with heavy:
            image = side_preview(store, tile, identity(photo_id, "Photo ID"), selected)
        return image_response(image)

    @api.post("/specimens/<specimen_id>/viewer/pair")
    def viewer_pair(specimen_id):
        data = json_request()
        tile = store.tile(specimen_id)
        with heavy:
            image, diagnostics = viewing_pair(store, tile, data)
        result = image_response(image)
        result.headers["X-WeaveDesk-Pair"] = json.dumps(diagnostics, separators=(",", ":"))
        return result

    @api.post("/specimens/<specimen_id>/quality")
    def quality(specimen_id):
        data = json_request()
        with heavy:
            report, _ = quality_manifest(store, specimen_id, data)
        return jsonify(report)

    @api.post("/specimens/<specimen_id>/quality/overlay")
    def quality_overlay(specimen_id):
        data = json_request()
        with heavy:
            _, image = quality_manifest(store, specimen_id, data)
        return image_response(image)

    @api.post("/specimens/<specimen_id>/regions")
    def region_add(specimen_id):
        return jsonify(present_tile(save_region(store, specimen_id, json_request()))), 201

    @api.put("/specimens/<specimen_id>/regions/<region_id>")
    def region_change(specimen_id, region_id):
        return jsonify(present_tile(save_region(store, specimen_id, json_request(), region_id)))

    @api.delete("/specimens/<specimen_id>/regions/<region_id>")
    def region_remove(specimen_id, region_id):
        return jsonify(present_tile(remove_region(store, specimen_id, region_id, json_request())))

    @api.put("/specimens/<specimen_id>/calibration")
    def calibration_save(specimen_id):
        with heavy:
            entry = calibrate(store, specimen_id, json_request())
        return jsonify(present_tile(entry))

    @api.delete("/specimens/<specimen_id>/calibration")
    def calibration_remove(specimen_id):
        return jsonify(present_tile(clear_calibration(store, specimen_id, json_request())))

    @api.post("/specimens/<specimen_id>/survey/preview")
    def survey_preview_route(specimen_id):
        tile = store.tile(specimen_id)
        with heavy:
            preview = survey_preview(store, tile, json_request())
        response = image_response(preview["image"])
        response.headers["X-Survey-Summary"] = json.dumps(preview["summary"], separators=(",", ":"))
        return response

    @api.put("/specimens/<specimen_id>/survey")
    def survey_save(specimen_id):
        with heavy:
            updated, summary = save_survey_plan(store, specimen_id, json_request())
        response = present_tile(updated)
        response["survey_summary"] = summary
        return jsonify(response)

    @api.delete("/specimens/<specimen_id>/survey")
    def survey_clear(specimen_id):
        return jsonify(present_tile(clear_survey_plan(store, specimen_id, json_request())))


def register_measurements(api, store, heavy):
    @api.get("/specimens/<specimen_id>/results")
    def measurements(specimen_id):
        return jsonify({"items": [summarize_result(item) for item in store.results(specimen_id)]})

    @api.post("/specimens/<specimen_id>/results")
    def measurement_add(specimen_id):
        with heavy:
            record = evaluate(store, specimen_id, json_request())
        return jsonify(record), 201

    @api.get("/results/<measurement_id>")
    def measurement_show(measurement_id):
        return jsonify(store.result(measurement_id))

    @api.get("/results/<measurement_id>/image/<layer>")
    def measurement_image(measurement_id, layer):
        data = store.result(measurement_id)
        chosen = enumeration(layer, ("overlay", "source", "paired", "reference"), "Measurement image")
        digest = data["metrics"].get("reference_image") if chosen == "reference" else data["pictures"][chosen]
        return png_blob(store, digest)

    @api.get("/results/<measurement_id>/metrics.csv")
    def measurement_csv(measurement_id):
        data = store.result(measurement_id)
        name = filename_stem(data["label"], "measurement")
        return binary_response(result_metrics_table(data), "text/csv", name + ".csv")

    @api.put("/results/<measurement_id>/observations/<entry>")
    def defect_decision(measurement_id, entry):
        data = json_request()
        revision = model_version(data)
        decision = flag_record_fields(data)
        with heavy:
            record = save_reviews(store, measurement_id, revision, {entry: decision})
        return jsonify(record)

    @api.get("/results/<measurement_id>/observations")
    def defect_list(measurement_id):
        mode = enumeration(request.args.get("decision", "all"), ("all", "review", "confirm", "dismiss"), "Review filter")
        phrase = string(request.args.get("query"), "Observation filter", maximum=100, required=False)
        return jsonify(review_selection(store.result(measurement_id), mode, phrase))

    @api.patch("/results/<measurement_id>/observations")
    def defect_review_many(measurement_id):
        data = json_request()
        choices = parse_reviews(data)
        with heavy:
            record = save_reviews(store, measurement_id, model_version(data), choices)
        return jsonify(record)

    @api.get("/library")
    def references():
        include_retired = request.args.get("archived") == "1"
        return jsonify({"items": store.references(include_retired)})

    @api.post("/library")
    def reference_save():
        with heavy:
            ref = approve_reference(store, json_request())
        return jsonify(ref), 201

    @api.post("/library/<reference_id>/retire")
    def reference_retire(reference_id):
        version = model_version(json_request())
        return jsonify(store.retire_reference(reference_id, version))

    @api.get("/library/<reference_id>/image")
    def ref_image(reference_id):
        return png_blob(store, store.reference(reference_id)["annotated_digest"])


def register_artifacts(api, store, heavy):
    @api.get("/specimens/<specimen_id>/evidence")
    def packets(specimen_id):
        return jsonify({"items": [summarize_evidence(item) for item in store.list_evidence(specimen_id)]})

    @api.post("/specimens/<specimen_id>/evidence")
    def packet_save(specimen_id):
        with heavy:
            item = create_evidence(store, specimen_id, json_request())
        return jsonify(summarize_evidence(item)), 201

    @api.get("/evidence/<packet_id>")
    def packet_info(packet_id):
        return jsonify(store.evidence(packet_id))

    @api.get("/evidence/<packet_id>/sheet/<int:number>")
    def packet_image(packet_id, number):
        record = store.evidence(packet_id)
        if not 1 <= number <= len(record["sheets"]):
            raise MissingError("Evidence sheet not found.")
        return png_blob(store, record["sheets"][number - 1])

    @api.get("/evidence/<packet_id>/inspection.html")
    def packet_html(packet_id):
        entry = store.evidence(packet_id)
        return binary_response(specimen_html(store, entry), "text/html")

    @api.get("/evidence/<packet_id>/archive.zip")
    def packet_zip(packet_id):
        entry = store.evidence(packet_id)
        with heavy:
            binary = evidence_zip(store, entry)
        name = filename_stem(entry["title"], "inspection")
        return binary_response(binary, "application/zip", name + ".zip")

    @api.get("/studies/measurements")
    def study_available():
        surface = {tile["id"]: tile for tile in store.list_tiles()}
        records = []
        allowed = set(public_metrics())
        for tile_id in surface:
            records.extend(summarize_result(row) for row in store.results(tile_id) if row["kind"] in allowed)
        return jsonify({"items": records})

    @api.get("/studies")
    def study_history():
        return jsonify({"items": store.studies()})

    @api.post("/studies")
    def study_make():
        with heavy:
            report = create_study(store, json_request())
        return jsonify(report), 201

    @api.get("/studies/<study_id>/sheet/<int:number>")
    def study_image(study_id, number):
        study = store.study(study_id)
        if not 1 <= number <= len(study["pictures"]):
            raise MissingError("Study sheet not found.")
        return png_blob(store, study["pictures"][number - 1])

    @api.get("/studies/<study_id>/data.<extension>")
    def study_files(study_id, extension):
        study = store.study(study_id)
        name = filename_stem(study["title"], "firing-study")
        selected = enumeration(extension, ("csv", "json"), "Study format")
        if selected == "csv":
            return binary_response(study_table(study), "text/csv", name + ".csv")
        return binary_response(artifact_json(study), "application/json", name + ".json")


def register_desks(api, store, heavy):
    @api.get("/received")
    def received_list():
        return jsonify({"items": store.received_packets()})

    @api.post("/received")
    def received_upload():
        incoming = request.files.get("packet")
        if incoming is None:
            raise InputError("Choose an evidence ZIP to check.")
        raw = incoming.stream.read(MAX_ARCHIVE_BYTES + 1)
        with heavy:
            entry, created = inspect_packet(store, raw)
        return jsonify(entry), 201 if created else 200

    @api.get("/received/<received_id>/sheet/<int:number>")
    def received_sheet(received_id, number):
        entry = store.received(received_id)
        if not 1 <= number <= len(entry["sheets"]):
            raise MissingError("Received evidence sheet not found.")
        return png_blob(store, entry["sheets"][number - 1])

    @api.get("/received/<received_id>/measurement/<entry_id>/<layer>")
    def received_measurement(received_id, entry_id, layer):
        packet = store.received(received_id)
        measurement = match_slot(packet["measurements"], identity(entry_id, "Measurement ID"), "measurement")
        image = enumeration(layer, ("source", "overlay", "paired"), "Measurement layer")
        return png_blob(store, measurement["pictures"].get(image))

    @api.get("/specimens/<specimen_id>/metrology")
    def geometry_records(specimen_id):
        return jsonify({"items": store.list_metrology(specimen_id)})

    @api.post("/specimens/<specimen_id>/metrology")
    def geometry_make(specimen_id):
        with heavy:
            record = create_metrology(store, specimen_id, json_request())
        return jsonify(record), 201

    @api.get("/metrology/<entry_id>/overlay.png")
    def geometry_artifact(entry_id):
        return png_blob(store, store.metrology(entry_id)["image"])

    @api.get("/metrology/<entry_id>/data.json")
    def geometry_document(entry_id):
        record = store.metrology(entry_id)
        name = filename_stem(record["title"], "geometry")
        return binary_response(artifact_json(record), "application/json", name + ".json")


def register_training(api, store, heavy):
    @api.get("/training/hero.png")
    def synthetic_thumbnail():
        return binary_response(preview_photo(), "image/png")

    @api.post("/training")
    def training():
        with heavy:
            tiles, created = load_training(store)
        return jsonify({"items": [summarize_tile(tile) for tile in tiles], "created": created}), 201 if created else 200


def create_app(data_dir=None):
    app = Flask(__name__, static_folder=None)
    chosen = Path(data_dir) if data_dir else Path(os.environ.get("WEAVEDESK_DATA", "data"))
    store = Store(chosen.resolve())
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD + 512 * 1024
    app.config["MAX_FORM_MEMORY_SIZE"] = 128 * 1024
    heavy = BoundedSemaphore(2)
    api = Blueprint("glaze_api", __name__, url_prefix="/api")
    app.extensions["weavedesk.store"] = store

    @app.get("/")
    def landing():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def assets(name):
        return send_from_directory(STATIC, name)

    @api.get("/health")
    def health():
        with store.connect() as conn:
            conn.execute("SELECT 1").fetchone()
        return jsonify({"status": "ok", "service": "weavedesk"})

    @api.get("/configuration")
    def configuration():
        return jsonify({
            "photo_roles": ROLE_LABELS,
            "region_purposes": REGION_PURPOSES,
            "analysis_labels": ANALYSIS_LABELS,
            "tools": TOOLS,
            "study_metrics": public_metrics(),
        })

    @app.errorhandler(InputError)
    def invalid(error):
        return error_response(error, 422)

    @app.errorhandler(BadRequest)
    def malformed(error):
        return error_response("The request could not be parsed.", 400)

    @app.errorhandler(MissingError)
    def missing(error):
        return error_response(error, 404)

    @app.errorhandler(ConflictError)
    def stale(error):
        return error_response(error, 409)

    @app.errorhandler(RequestEntityTooLarge)
    def oversize(error):
        return error_response("Upload is larger than the request limit.", 413)

    @app.after_request
    def local_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if request.path.startswith("/api/") and response.mimetype == "application/json":
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    register_training(api, store, heavy)
    register_specimens(api, store, heavy)
    register_measurements(api, store, heavy)
    register_artifacts(api, store, heavy)
    register_desks(api, store, heavy)
    app.register_blueprint(api)
    return app
