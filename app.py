import io
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove
from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

app = Flask(__name__)
CORS(app)

# Haar cascade para sa face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_and_crop_face(pil_img, target_size, face_ratio=0.60, offset=0.0):
    """Detect face, align eyes on 1/3 line + offset, crop properly."""
    cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return ImageOps.fit(pil_img, target_size, method=Image.LANCZOS)

    # Largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    cx, cy = x + w // 2, y + h // 2

    eye_level = y + int(h * 0.25)
    desired_face_height = int(target_size[1] * face_ratio)
    if desired_face_height < 40:
        desired_face_height = 40

    scale = desired_face_height / h
    crop_w = int(target_size[0] / scale)
    crop_h = int(target_size[1] / scale)

    target_eye_y = int(crop_h * (1/3))
    target_eye_y = int(target_eye_y + (offset * crop_h))

    y1 = max(0, eye_level - target_eye_y)
    x1 = max(0, cx - crop_w // 2)
    x2 = min(cv_img.shape[1], x1 + crop_w)
    y2 = min(cv_img.shape[0], y1 + crop_h)

    cropped = pil_img.crop((x1, y1, x2, y2))
    return ImageOps.fit(cropped, target_size, method=Image.LANCZOS, centering=(0.5, 0.5))


def apply_background(image_bytes, bg_color):
    """Remove background and apply new one."""
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    result = remove(input_image)

    if bg_color is not None:
        bg = Image.new("RGBA", result.size, bg_color + (255,))
        result = Image.alpha_composite(bg, result)

    return result.convert("RGB")


@app.route("/remove", methods=["POST"])
def remove_bg_batch():
    files = request.files.getlist("images")
    bg_color = request.form.get("bg_color", "transparent")
    face_ratio = float(request.form.get("face_ratio", 0.60))
    offset = float(request.form.get("offset", 0.0))
    width_in = float(request.form.get("width_in", 1.77))
    height_in = float(request.form.get("height_in", 1.5))

    if not files:
        return jsonify({"error": "No images uploaded."}), 400

    if bg_color == "transparent":
        bg_tuple = None
    else:
        try:
            if bg_color.startswith("#") and len(bg_color) == 7:
                bg_tuple = tuple(int(bg_color[i:i+2], 16) for i in (1,3,5))
            else:
                bg_tuple = (255,255,255)
        except Exception:
            bg_tuple = (255,255,255)

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    page_w, page_h = A4

    dpi = 300
    cell_w, cell_h = int(width_in * dpi), int(height_in * dpi)
    x_margin, y_margin = 20, 20
    spacing_x, spacing_y = 20, 20
    x, y = x_margin, page_h - cell_h - y_margin

    for file in files:
        try:
            result_image = apply_background(file.read(), bg_tuple)
            fitted = detect_and_crop_face(result_image, (cell_w, cell_h), face_ratio, offset)

            img_buffer = io.BytesIO()
            fitted.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            img_reader = ImageReader(img_buffer)

            # Draw image
            c.drawImage(img_reader, x, y, width=cell_w*0.24, height=cell_h*0.24)

            # Draw border
            c.setLineWidth(0.75)
            c.setStrokeColor(colors.black)
            c.rect(x, y, cell_w*0.24, cell_h*0.24)

            # Move to next grid
            x += cell_w*0.25
            if x + cell_w*0.24 > page_w - x_margin:
                x = x_margin
                y -= cell_h*0.25
                if y < y_margin:
                    c.showPage()
                    x, y = x_margin, page_h - cell_h - y_margin

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")

    # Optional: Draw grid on full page for cutting guide
    c.setLineWidth(0.5)
    c.setStrokeColor(colors.grey)
    for gx in range(x_margin, int(page_w-x_margin), int(cell_w*0.25+spacing_x)):
        c.setDash(3,3)
        c.line(gx, y_margin, gx, page_h - y_margin)
    for gy in range(y_margin, int(page_h-y_margin), int(cell_h*0.25+spacing_y)):
        c.line(x_margin, gy, page_w - x_margin, gy)
    c.setDash()

    c.save()
    pdf_buffer.seek(0)

    return send_file(pdf_buffer, as_attachment=True, download_name="id_photos.pdf", mimetype="application/pdf")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
