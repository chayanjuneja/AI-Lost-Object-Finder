def get_zone_from_bbox(x1, y1, x2, y2, frame_w, frame_h):
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    if mid_x < frame_w / 3:
        x_zone = "left side"
    elif mid_x > 2 * frame_w / 3:
        x_zone = "right side"
    else:
        x_zone = "center"

    if mid_y < frame_h / 2:
        y_zone = "upper"
    else:
        y_zone = "lower"

    return f"{y_zone} {x_zone}"
