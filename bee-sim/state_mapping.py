import math

def bbox_to_state(x1, y1, x2, y2, img_w, img_h, n_bins=10):
    """
    Returns:
      state_id in [0, 29] where:
        region_id ∈ {0:left, 1:center, 2:right}
        closeness_bin ∈ {0..9}
        state = region_id * 10 + closeness_bin
    """
    # center of bbox
    cx = (x1 + x2) / 2.0

    # region split (3 vertical sections)
    if cx < img_w / 3:
        region_id = 0
    elif cx < 2 * img_w / 3:
        region_id = 1
    else:
        region_id = 2

    # closeness as bbox_area / image_area
    bbox_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    closeness = bbox_area / float(img_w * img_h)  # 0..1 (usually small)

    # bin it into 10 ranges
    # Example: bin 0 = [0,0.1), bin 9 = [0.9,1.0]
    bin_id = int(math.floor(closeness * n_bins))
    if bin_id < 0:
        bin_id = 0
    if bin_id >= n_bins:
        bin_id = n_bins - 1

    state = region_id * n_bins + bin_id
    return state, region_id, bin_id, closeness
