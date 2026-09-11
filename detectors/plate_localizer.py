"""Find plate-like text regions in a frame using EasyOCR bounding boxes."""

def detect_and_crop_plate(frame, reader):
    """Return a padded crop around the most plate-like text region, or None."""
    results = reader.readtext(frame, detail=1)
    if not results:
        return None

    height, width = frame.shape[:2]
    best_crop = None
    best_score = 0.0

    for bbox, _text, confidence in results:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        x1 = int(max(0, min(xs)))
        y1 = int(max(0, min(ys)))
        x2 = int(min(width, max(xs)))
        y2 = int(min(height, max(ys)))

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < 20 or box_height < 10:
            continue

        aspect_ratio = box_width / box_height
        if aspect_ratio < 1.5 or aspect_ratio > 8.0:
            continue

        area_ratio = (box_width * box_height) / (width * height)
        score = float(confidence) * min(area_ratio * 50, 1.0)
        if score <= best_score:
            continue

        pad_x = int(box_width * 0.1)
        pad_y = int(box_height * 0.2)
        crop = frame[
            max(0, y1 - pad_y) : min(height, y2 + pad_y),
            max(0, x1 - pad_x) : min(width, x2 + pad_x),
        ]
        if crop.size == 0:
            continue

        best_score = score
        best_crop = crop

    return best_crop
